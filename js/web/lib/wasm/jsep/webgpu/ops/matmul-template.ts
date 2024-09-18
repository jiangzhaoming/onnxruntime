// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {DataType} from '../../../wasm-common';
import { LOG } from '../../log';
import {TensorView} from '../../tensor-view';
import {/*BroadcastUtil,*/ ShapeUtil} from '../../util';
import {/*ComputeContext,*/ ProgramInfo, ProgramUniform} from '../types';

import {createTensorShapeVariables, getBroadcastDims, getMaxComponents, IndicesHelper, inputVariable, internalVariable, outputVariable, ShaderHelper, tensorTypeToWsglStorageType, UniformsArrayType} from './common';
import {appendActivationUniforms, appendActivationUniformsData, getActivationSnippet, InternalActivationAttributes} from './fuse-utils';

// GEMM: Out[MxN] = activation(A[MxK] * B[KxN] + Bias[MxN])
interface BufferLayoutInfo {
  // vector packing direction alone logical direction, 'row' for row vector containing multiple cols (1xN)
  // and 'col' for col vector containing multiple rows (Nx1).
  packed_vector_direction: 'row'|'col';
  packed_vector_size: 1|2|4;
  // Used to generate buffer accessing helper function. NWH layout is transposed.
  loading_points_layout: 'NHW'|'NWH';
  // Buffer accessing: buffer_variable[batch][inner][outer] of packed type
  buffer_inner_boundary: number;
  buffer_outer_boundary: number;
}

interface TensorInfo {
  // Aggregated batches number
  aggregated_batches: number;
  // Logical rows and cols
  rows: number;
  cols: number;
  type: 'f32'|'f16';
  // TensorView, should be of 3-dims shape
  // tensor_view?: TensorView;
  // Physical buffer
  // Expected buffer variable: buffer_variable[batch][inner][outer] of packed type
  // buffer_variable: IndicesHelper;
  buffer_layout: BufferLayoutInfo;
}

// Original batch dims information, used for handling broadcasting
interface BatchesInfo {
  output_batch_dims: readonly number[];
  input_A_batch_dims: readonly number[];
  input_B_batch_dims: readonly number[];
}

interface GEMMOperationParameters {
  // batches: number;
  batches_info: BatchesInfo;
  tensor_data_type: number;
  input_A: TensorInfo;
  input_B: TensorInfo;
  input_Bias?: TensorInfo;
  output: TensorInfo;
  activation: InternalActivationAttributes;
}

/*
interface PerWorkgroupLimits {
    maximium_register_size: number;
    maximium_shared_memory_size: number;
};
*/

// No data communication betweens workgroups.
interface GEMMWorkgroupScheduleParameters {
  output_rows_per_workgroup: number;
  output_cols_per_workgroup: number;
  threads_per_workgroup: number;
  // limits: PerWorkgroupLimits;
}

/*
type ScheduleSchemaSubgroupsParameters =
    {
        use_subgroups: false;
    } | {
        use_subgroups: true;
        subgroups_shared_on: 'A' | 'B';
        // Consider adaptive subgroup size
        subgroup_size: number;
    };
*/

// Tensor slice divides inputs along K dimension,
// makes A[MxK]*B[KxN] into Ai[MxKs]*Bi[KsxN], Ks = K / factor,
// i = 0 to factor-1.
type ScheduleSchemaTensorSliceParameters = {
  tensor_slice_factor: number;
  // How to divide the input A and B along K dimension
  tensor_slice_input_policy: 'continious' | 'interleaved';
};

type ShaderScheduleSchemaParameters = {
  compute_schema: 'dotProduct'|'scaledVector'; tensor_slice: ScheduleSchemaTensorSliceParameters;
};

// const arrayOf = (length: number) => Array.from({length});
const arrayMap = <Type>(length: number, func: (i: number) => Type) => Array.from({length}).map((_, i) => func(i));

const assert =
    (condition: boolean, message: string) => {
      if (!condition) {
        throw new Error(message);
      }
    }

const addIdent = (code: string, width: number, first_line_hint: 'keepFirstLine'|'addAllLines' = 'addAllLines') => {
  if (width === 0) {
    return code;
  }
  const ident = ' '.repeat(width);
  return (
      code.split('\n')
          .map((line, i) => (line === '' || ((first_line_hint === 'keepFirstLine') && (i === 0))) ? line : (ident + line))
          .join('\n')  //
  );
};

const forWGSLBuilder = (    //
    loop_var_name: string,  //
    initializer: {WGSL: (loop_var: string) => string, symbolic?: () => number},
    loop_condition: {WGSL: (loop_var: string) => string, symbolic?: (loop_value: number) => boolean},
    loop_update: {WGSL: (loop_var: string) => string, symbolic?: (loop_value: number) => number},
    loop_body: (loop_var_or_value: string|number) => string,  //
    outer_ident_width: number = 0,                            //
    inner_loop_add_ident_width: number = 0) => {
  const outer_ident = ' '.repeat(outer_ident_width);
  if ((typeof initializer.symbolic !== 'undefined') && (typeof loop_condition.symbolic !== 'undefined') &&
      (typeof loop_update.symbolic !== 'undefined')) {
    // All symbolic information is provided, do the symbolic loop unfold
    let unfolded_loop_bodies: string[] = [];
    for (let loop_value: number = initializer.symbolic(); loop_condition.symbolic(loop_value);
         loop_value = loop_update.symbolic(loop_value)) {
      unfolded_loop_bodies.push(`
${outer_ident}{
${addIdent(loop_body(loop_value), inner_loop_add_ident_width, 'addAllLines')}
${outer_ident}}`);
    }
    return unfolded_loop_bodies.join('\n');
  } else {
    return `
${outer_ident}for (${initializer.WGSL(loop_var_name)}; ${loop_condition.WGSL(loop_var_name)}; ${
        loop_update.WGSL(loop_var_name)}) {
${addIdent(loop_body(loop_var_name), inner_loop_add_ident_width, 'addAllLines')}
${outer_ident}}`
  }
};

const integerLoopUpFrom0WGSL = (
    loop_var_name: string,
    loop_var_type_WGSL: 'u32'|'i32',
    upper_boundary: number|string,  // cond: loop_var_or_value < upper_boundary
    loop_body: (loop_var_or_value: string|number) => string,
    outer_ident_width: number = 0,
    ) =>
    forWGSLBuilder(
        loop_var_name,
        {
          symbolic: () => 0,  //
          WGSL: (loop_var: string) => `var ${loop_var}: ${loop_var_type_WGSL} = 0`
        },
        {
          WGSL: (loop_var: string) => `${loop_var} < ${loop_var_type_WGSL}(${upper_boundary})`,
          ...(typeof upper_boundary === 'number' ? {symbolic: (loop_value: number) => loop_value < upper_boundary} : {})
        },
        {
          WGSL: (loop_var: string) => `${loop_var}++`,  //
          symbolic: (loop_value: number) => loop_value + 1
        },
        loop_body,         //
        outer_ident_width  //
    );

const u32LoopUpFrom0WGSL = (
    loop_var_name: string,
    upper_boundary: number|string,  // cond: loop_var_or_value < upper_boundary
    loop_body: (loop_var_or_value: string|number) => string,
    outer_ident_width: number = 0,
    ) => integerLoopUpFrom0WGSL(loop_var_name, 'u32', upper_boundary, loop_body, outer_ident_width);

interface RowsColsSpan {
  rows: number;
  cols: number;
}

interface MKNSpan {
  M: number;
  K: number;
  N: number;
}

const PrintMKN = (a: MKNSpan) => `{M: ${a.M}, K: ${a.K}, N: ${a.N}}`;
const MKNDivisible = (a: MKNSpan, b: MKNSpan) => ((a.M % b.M === 0) && (a.K % b.K === 0) && (a.N % b.N === 0));
const MKNDivision = (a: MKNSpan, b: MKNSpan) => ({M: a.M / b.M, K: a.K / b.K, N: a.N / b.N} as MKNSpan);

const generateVectorizedComputeStatements = (
    inputAVectorWGSLs: string[/* row groups */][/* col groups */],
    inputBVectorWGSLs: string[/* row groups */][/* col groups */],
    outputPackedWGSLName: string /* Must be a 2d array of vec/sclr type */|string[][],
    outputSpanUpperLeftPositionPackedRow: string|number,
    outputSpanUpperLeftPositionPackedCol: string|number,
    scalarTypeWGSL: 'f32'|'f16',
    params: {
      inputAVectorSize: 1|2|4;               //
      inputAVectorDirection: 'row' | 'col';  //
      inputBVectorSize: 1 | 2 | 4;           //
      inputBVectorDirection: 'row' | 'col';
      outputVectorSize: 1 | 2 | 4;
      computeSchema: 'dotProduct' | 'scaledVector';
      // Assume output vector is row vector.
      // Assume outputColGroups = 1
      expectedOutputRowGroups: 1 | 2 | 4;
    },
    ) => {
  const {
    inputAVectorSize,
    inputAVectorDirection,
    inputBVectorSize,
    inputBVectorDirection,
    outputVectorSize,
    computeSchema,
    expectedOutputRowGroups
  } = params;
  const inputARowGroups = inputAVectorWGSLs.length;
  const inputAColGroups = inputAVectorWGSLs[0].length;
  const inputBRowGroups = inputBVectorWGSLs.length;
  const inputBColGroups = inputBVectorWGSLs[0].length;
  const inputAScalarRows = ((inputAVectorDirection === 'row') ? 1 : inputAVectorSize) * inputARowGroups;
  const inputAScalarCols = inputAColGroups * ((inputAVectorDirection === 'row') ? inputAVectorSize : 1);
  const inputBScalarRows = inputBRowGroups * ((inputBVectorDirection === 'col') ? inputBVectorSize : 1);
  const inputBScalarCols = ((inputBVectorDirection === 'row') ? inputBVectorSize : 1) * inputBColGroups;
  const outputSpanRow = inputAScalarRows;
  const outputSpanCol = inputBScalarCols;
  assert(
      outputSpanCol === outputVectorSize,
      `Expect just enough input B vectors to fill output vector size ${outputVectorSize}, got ${outputSpanCol}.`);
  assert(
      inputAScalarCols === inputBScalarRows,
      `Expect equal size along K dimension, got A scalar cols ${inputAScalarCols} and B scalar rows ${
          inputBScalarRows}.`);
  assert(
      outputSpanRow === expectedOutputRowGroups,
      `Expect compute block of M=${expectedOutputRowGroups}, got ${outputSpanRow}`);
  const outputPackedTypeWGSL =
      outputVectorSize === 1 ? `${scalarTypeWGSL}` : `vec${outputVectorSize}<${scalarTypeWGSL}>`;

  const composeScalarsToOutputVector = (scalars: string[]) => {
    assert(scalars.length === outputVectorSize, `Expect ${outputVectorSize} output components, got ${scalars.length}`);
    return `${outputPackedTypeWGSL}(${scalars.join(', ')})`;
  };

  const inputScalarAccessorBuilder = (inputVectorWGSLs: string[][], inputVectorSize: 1|2|4) =>
      ((inputVectorSize === 1) ?
           ((inputRowGroupId: number, inputColGroupId: number, inputVectorElement: number) => {
             if (inputVectorElement !== 0) {
               throw (`Accessing unexpected component ${inputVectorElement} on scalar type input`);
             }
             return `${inputVectorWGSLs[inputRowGroupId][inputColGroupId]}`;
           }) :
           ((inputRowGroupId: number, inputColGroupId: number, inputVectorElement: number) => {
             return `${inputVectorWGSLs[inputRowGroupId][inputColGroupId]}.${'xyzw'[inputVectorElement]}`;
           }));
  const inputAScalarAccessor = inputScalarAccessorBuilder(inputAVectorWGSLs, inputAVectorSize);
  const inputBScalarAccessor = inputScalarAccessorBuilder(inputBVectorWGSLs, inputBVectorSize);
  /*
  const inputAScalarAccessor = (inputARowGroupId: number, inputAColGroupId: number, inputAVectorElement: number) => {
    if (inputAVectorSize === 1) {
      if ((typeof inputAVectorElement === 'number') && (inputAVectorElement !== 0)) {
        throw (`Accessing unexpected component ${inputAVectorElement} on scalar type input A`);
      }
    }
    return `${inputAVectorWGSLs[inputARowGroupId][inputAColGroupId]}${
        inputAVectorSize === 1 ? '' : `.${'xyzw'[inputAVectorElement]}`}`;
  };
  const inputBScalarAccessor = (inputBRowGroupId: number, inputBColGroupId: number, inputBVectorElement: number) => {
    if (inputBVectorSize === 1) {
      if ((typeof inputBVectorElement === 'number') && (inputBVectorElement !== 0)) {
        throw (`Accessing unexpected component ${inputBVectorElement} on scalar type input B`);
      }
    }
    return `${inputBVectorWGSLs[inputBRowGroupId][inputBColGroupId]}${
        inputBVectorSize === 1 ? '' : `.${'xyzw'[inputBVectorElement]}`}`;
  };
  */

  const inputAScalarPositionAccessor = (scalarRow: number, scalarCol: number) => (inputAVectorDirection === 'row') ?
      `${inputAScalarAccessor(scalarRow, Math.floor(scalarCol / inputAVectorSize), scalarCol % inputAVectorSize)}` :
      `${inputAScalarAccessor(Math.floor(scalarRow / inputAVectorSize), scalarCol, scalarRow % inputAVectorSize)}`;
  const inputBScalarPositionAccessor = (scalarRow: number, scalarCol: number) => (inputBVectorDirection === 'row') ?
      `${inputBScalarAccessor(scalarRow, Math.floor(scalarCol / inputBVectorSize), scalarCol % inputBVectorSize)}` :
      `${inputBScalarAccessor(Math.floor(scalarRow / inputBVectorSize), scalarCol, scalarRow % inputBVectorSize)}`;

  const biasedOutputPackedRow = (bias: string|number) =>
      ((typeof outputSpanUpperLeftPositionPackedRow === 'number') && (typeof bias === 'number')) ?
      bias + outputSpanUpperLeftPositionPackedRow :
      `${bias}+${outputSpanUpperLeftPositionPackedRow}`;
  const outputPackedRowBiasedAccessor = (outputRowBias: string|number) => {
    if (typeof outputPackedWGSLName === 'string') {
      return `${outputPackedWGSLName}[${biasedOutputPackedRow(outputRowBias)}][${
          outputSpanUpperLeftPositionPackedCol}]`;
    } else {
      assert(
          typeof outputRowBias === 'number',
          `Expect outputRowBias of number type for outputRowBias[][], got ${typeof outputRowBias}`);
      assert(
          typeof outputSpanUpperLeftPositionPackedRow === 'number',
          `Expect outputSpanUpperLeftPositionPackedRow of number type for outputRowBias[][], got ${
              typeof outputSpanUpperLeftPositionPackedRow}`);
      assert(
          typeof outputSpanUpperLeftPositionPackedCol === 'number',
          `Expect outputSpanUpperLeftPositionPackedCol of number type for outputRowBias[][], got ${
              typeof outputSpanUpperLeftPositionPackedCol}`);
      const row = (outputRowBias as number) + (outputSpanUpperLeftPositionPackedRow as number);
      const col = outputSpanUpperLeftPositionPackedCol as number;
      assert(
          row < outputPackedWGSLName.length && col < outputPackedWGSLName[row].length,
          `Accessing outputPackedWGSLName[${row}][${col}] out of bound`);
      return outputPackedWGSLName[row][col];
    }
  };

  const dotOrSalarMul = (a: string, b: string, size: number) => size === 1 ? `(${a}*${b})` : `dot(${a}, ${b})`;

  const computeDotProdAmongK = (outputScalarRow: number, outputScalarCol: number) => {
    const kDimensionSize = inputAScalarCols;
    if (kDimensionSize < 2) {
      throw (`K dimension size ${kDimensionSize} is too small`);
    }
    const gatherRowVectorFromInputA = (scalarRow: number, scalarCol: number, vectorSize: 1|2|4) => {
      if ((scalarRow >= inputAScalarRows) || (scalarCol + vectorSize > kDimensionSize)) {
        throw (`gatherRowVectorFromInputA OOB, requiring [${scalarRow}][${scalarCol}:+${
            vectorSize}] on input A with scalars [${inputAScalarRows}][${kDimensionSize}]`);
      }
      return `${vectorSize === 1 ? scalarTypeWGSL : `vec${vectorSize}<${scalarTypeWGSL}>`}(${
          arrayMap(vectorSize, (col) => `${inputAScalarPositionAccessor(scalarRow, scalarCol + col)}`).join(', ')})`;
    };
    const gatherColVectorFromInputB = (scalarRow: number, scalarCol: number, vectorSize: 1|2|4) => {
      if ((scalarRow + vectorSize > kDimensionSize) || (scalarCol > inputBScalarCols)) {
        throw (`gatherColVectorFromInputB OOB, requiring [${scalarRow}:+${vectorSize}][${
            scalarCol}] on input B with scalars [${kDimensionSize}][${inputBScalarCols}]`);
      }
      return `${vectorSize === 1 ? scalarTypeWGSL : `vec${vectorSize}<${scalarTypeWGSL}>`}(${
          arrayMap(vectorSize, (row) => `${inputBScalarPositionAccessor(scalarRow + row, scalarCol)}`).join(', ')})`;
    };
    // const dotProdStepSize = (kDimensionSize % 4 === 0)?4:((kDimensionSize % 2 === 0)?2:1);
    const kDimensionSizeOf4 = Math.floor(kDimensionSize / 4) * 4;
    const kDimensionSizeOf2 = Math.floor((kDimensionSize - kDimensionSizeOf4) / 2) * 2;
    const kDimensionSizeOf1 = kDimensionSize - kDimensionSizeOf4 - kDimensionSizeOf2;
    return [
      ...arrayMap(
          kDimensionSizeOf4 / 4,
          (step4) => dotOrSalarMul(
              gatherRowVectorFromInputA(outputScalarRow, step4 * 4, 4),
              gatherColVectorFromInputB(step4 * 4, outputScalarCol, 4), 4)),
      ...arrayMap(
          kDimensionSizeOf2 / 2,
          (step2) => dotOrSalarMul(
              gatherRowVectorFromInputA(outputScalarRow, kDimensionSizeOf4 + step2 * 2, 2),
              gatherColVectorFromInputB(kDimensionSizeOf4 + step2 * 2, outputScalarCol, 2), 2)),
      ...arrayMap(
          kDimensionSizeOf1,
          (step1) => dotOrSalarMul(
              gatherRowVectorFromInputA(outputScalarRow, kDimensionSizeOf4 + kDimensionSizeOf2 + step1, 1),
              gatherColVectorFromInputB(kDimensionSizeOf4 + kDimensionSizeOf2 + step1, outputScalarCol, 1), 1)),
    ].join(' + ');
  };

  if (computeSchema === 'dotProduct') {
    return (
        arrayMap(
            outputSpanRow,
            (outputRow) => `${outputPackedRowBiasedAccessor(outputRow)} += ${
                composeScalarsToOutputVector(
                    arrayMap(outputSpanCol, (outputCol) => computeDotProdAmongK(outputRow, outputCol)))};`)
            .join('\n')  //
    );
  } else {
    // Schema scaledVector
    const gatherRowVectorFromInputB = (scalarRow: number, scalarCol: number, vectorSize: 1|2|4) => {
      if ((scalarRow >= inputBScalarRows) || (scalarCol + vectorSize > inputBScalarCols)) {
        throw (`gatherRowVectorFromInputB OOB, requiring [${scalarRow}][${scalarCol}:+${
            vectorSize}] on input A with scalars [${inputBScalarRows}][${inputBScalarCols}]`);
      }
      return `${vectorSize === 1 ? scalarTypeWGSL : `vec${vectorSize}<${scalarTypeWGSL}>`}(${
          arrayMap(vectorSize, (col) => `${inputBScalarPositionAccessor(scalarRow, scalarCol + col)}`).join(', ')})`;
    };

    return (
        arrayMap(
            outputSpanRow,
            (outputRow) => `${outputPackedRowBiasedAccessor(outputRow)} += ${
                arrayMap(
                    inputAScalarCols,
                    (k) => `(${inputAScalarPositionAccessor(outputRow, k)} * ${
                        gatherRowVectorFromInputB(k, 0, outputVectorSize)})`)
                    .join(' + ')};`)
            .join('\n')  //
    );
  }
};

export function templatedMatMulProgram(
    op_params: GEMMOperationParameters,
    workgroup_params: GEMMWorkgroupScheduleParameters,
    schedule_params: ShaderScheduleSchemaParameters,
    ): ProgramInfo {
  const {batches_info, input_A, input_B, input_Bias, output, activation, tensor_data_type} = op_params;

  const {output_rows_per_workgroup, output_cols_per_workgroup} = workgroup_params;
  const {compute_schema, tensor_slice} = schedule_params;
  const {tensor_slice_factor} = tensor_slice;

  const has_bias = typeof input_Bias !== 'undefined';

  /*
  assert(
      output.aggregated_batches === input_A.aggregated_batches * input_B.aggregated_batches,
      `Expect output.batches ${output.aggregated_batches} === input_A.batches ${input_A.aggregated_batches} *
  input_B.batches ${ input_B.aggregated_batches}`);
  */
  assert(
      !has_bias || output.aggregated_batches === input_Bias.aggregated_batches,
      `Expect input_bias.batches ${input_Bias?.aggregated_batches} === output.batches ${output.aggregated_batches}`);

  assert(
      input_A.cols === input_B.rows,
      `Expect logical cols of input A (${input_A.cols}) is the same as logical rows of B (${input_B.rows})`);
  const logical_scalar_size = {M: input_A.rows, K: input_A.cols, N: input_B.cols};
  LOG(`fatal`, `logical_scalar_size: ${PrintMKN(logical_scalar_size)}`);

  // Each workgroup should go through the whole K dimension to get full result, tensor slice along K happens
  // within a workgroup. Note that logical_scalar_size might be undividable by
  // logical_scalar_size_per_workgroup.
  const logical_scalar_size_per_workgroup = {
    M: output_rows_per_workgroup,
    K: input_A.cols,
    N: output_cols_per_workgroup
  };

  const packing_shape_A = {
    rows: (input_A.buffer_layout.packed_vector_direction === 'col' ? input_A.buffer_layout.packed_vector_size : 1),
    cols: (input_A.buffer_layout.packed_vector_direction === 'row' ? input_A.buffer_layout.packed_vector_size : 1),
  };
  const packing_shape_B = {
    rows: (input_B.buffer_layout.packed_vector_direction === 'col' ? input_B.buffer_layout.packed_vector_size : 1),
    cols: (input_B.buffer_layout.packed_vector_direction === 'row' ? input_B.buffer_layout.packed_vector_size : 1),
  };
  const packing_shape_Output = {
    rows: (output.buffer_layout.packed_vector_direction === 'col' ? output.buffer_layout.packed_vector_size : 1),
    cols: (output.buffer_layout.packed_vector_direction === 'row' ? output.buffer_layout.packed_vector_size : 1),
  };
  assert(
      output_rows_per_workgroup % packing_shape_A.rows === 0,
      `Expect output_rows_per_workgroup (${output_rows_per_workgroup}) is a multiple of row packing size of A (${
          packing_shape_A.rows})`);
  assert(
      output_cols_per_workgroup % packing_shape_B.cols === 0,
      `Expect output_rows_per_workgroup (${output_cols_per_workgroup}) is a multiple of col packing size of B (${
          packing_shape_B.cols})`);
  assert(
      output_rows_per_workgroup % packing_shape_Output.rows === 0,
      `Expect output_rows_per_workgroup (${output_rows_per_workgroup}) is a multiple of row packing size of output (${
          packing_shape_Output.rows})`);
  assert(
      output_cols_per_workgroup % packing_shape_Output.cols === 0,
      `Expect output_rows_per_workgroup (${output_cols_per_workgroup}) is a multiple of col packing size of output (${
          packing_shape_Output.cols})`);
  assert(
      packing_shape_Output.cols % packing_shape_B.cols === 0,
      `Expect output cols packing size ${packing_shape_Output.cols} is a multiple of input B cols packing size ${
          packing_shape_B.cols}`);
  /*
  const loading_points_per_workgroup_A: RowsColsSpan = {
      rows: output_rows_per_workgroup / packing_shape_A.rows,
      cols: input_A.cols / packing_shape_A.cols,
  };
  const loading_points_per_workgroup_B: RowsColsSpan = {
      rows: input_B.rows / packing_shape_B.rows,
      cols: output_cols_per_workgroup / packing_shape_B.cols,
  };
  */
  const compute_block_shape: MKNSpan = {
    M: packing_shape_A.rows,
    K: Math.max(packing_shape_A.cols, packing_shape_B.rows),
    N: packing_shape_Output.cols,  // Must be a multiple of packing_shape_B.cols
  };
  assert(
      MKNDivisible(logical_scalar_size_per_workgroup, compute_block_shape),
      `Expect logical_scalar_size_per_workgroup ${PrintMKN(logical_scalar_size_per_workgroup)} \
dividable by compute_block_shape ${PrintMKN(compute_block_shape)}`);
  const compute_blocks_per_workgroup = MKNDivision(logical_scalar_size_per_workgroup, compute_block_shape);
  const loading_points_per_compute_block_A: RowsColsSpan = {
    rows: compute_block_shape.M / packing_shape_A.rows,  // 1 since compute_block_shape.M === packing_shape_A.rows
    cols: compute_block_shape.K / packing_shape_A.cols,
  };
  const loading_points_per_compute_block_B: RowsColsSpan = {
    rows: compute_block_shape.K / packing_shape_B.rows,
    cols: compute_block_shape.N / packing_shape_B.cols,
  };
  const loading_points_per_compute_block_Output: RowsColsSpan = {
    rows: compute_block_shape.M / packing_shape_Output.rows,
    cols: compute_block_shape.N / packing_shape_Output.cols,
  };
  LOG(`fatal`, `compute_block_shape: ${PrintMKN(compute_block_shape)}`);

  // -----------------------------------------------------------------------------
  //   Scheduling decision
  // -----------------------------------------------------------------------------
  // TODO: Should be decided by schedule.
  const compute_block_thread_K_inner_loop_step: number|string = 1;
  assert(
      workgroup_params.threads_per_workgroup % tensor_slice_factor === 0,
      `Workgroup size ${workgroup_params.threads_per_workgroup} should be a multiple of tensor_slice_factor ${
          tensor_slice_factor}`);
  const threads_per_TSG = workgroup_params.threads_per_workgroup / tensor_slice_factor;
  const threads_along_M_per_TSG = 16;
  assert(
      threads_per_TSG % threads_along_M_per_TSG === 0,
      `threads_per_TSG ${threads_per_TSG} should be a multiple of threads_along_M_per_TSG ${threads_along_M_per_TSG}`);
  const threads_along_N_per_TSG = threads_per_TSG / threads_along_M_per_TSG;
  assert(
      compute_blocks_per_workgroup.M % threads_along_M_per_TSG === 0,
      `compute_blocks_per_workgroup.M ${
          compute_blocks_per_workgroup.M} should be a multiple of threads_along_M_per_TSG ${threads_along_M_per_TSG}`);
  const compute_blocks_per_thread_M: number|string = compute_blocks_per_workgroup.M / threads_along_M_per_TSG;
  assert(
      compute_blocks_per_workgroup.N % threads_along_N_per_TSG === 0,
      `compute_blocks_per_workgroup.N ${
          compute_blocks_per_workgroup.N} should be a multiple of threads_along_N_per_TSG ${threads_along_N_per_TSG}`);
  const compute_blocks_per_thread_N: number|string = compute_blocks_per_workgroup.N / threads_along_N_per_TSG;

  const dispatch_workgroups = {
    x: Math.ceil(logical_scalar_size.M / logical_scalar_size_per_workgroup.M),
    y: Math.ceil(logical_scalar_size.N / logical_scalar_size_per_workgroup.N),
    z: output.aggregated_batches,
  } as const;

  // -----------------------------------------------------------------------------
  //   Handle the batch dims broadcasting
  // -----------------------------------------------------------------------------
  const {output_batch_dims, input_A_batch_dims, input_B_batch_dims} = batches_info;
  const input_A_broadcasted_dims = getBroadcastDims(input_A_batch_dims, output_batch_dims);
  const input_B_broadcasted_dims = getBroadcastDims(input_B_batch_dims, output_batch_dims);

  // -----------------------------------------------------------------------------
  //   Uniforms definition and values
  // -----------------------------------------------------------------------------
  // Uniform definition
  const uniforms_WGSL_info: UniformsArrayType = [
    // { name: 'input_A_batches', type: 'u32' },
    // {name: 'input_B_batches', type: 'u32'},
  ];
  appendActivationUniforms(activation, uniforms_WGSL_info);

  // Uniforms values
  const programUniforms: ProgramUniform[] = [
    // { /* input_A_batches */ type: DataType.uint32, data: input_A.batches },
    // {/* input_B_batches */ type: DataType.uint32, data: input_B.aggregated_batches},
  ];
  appendActivationUniformsData(activation, programUniforms);
  // Tensor shapes of all input/output variables must be pushed into programUniforms
  assert(
    input_A.buffer_layout.buffer_inner_boundary ===
        ((input_A.buffer_layout.loading_points_layout === 'NHW') ? logical_scalar_size.M : logical_scalar_size.K),
    `Expect for layout ${input_A.buffer_layout.loading_points_layout} \
input_A.buffer_layout.buffer_inner_boundary ${input_A.buffer_layout.buffer_inner_boundary} === \
${(input_A.buffer_layout.loading_points_layout === 'NHW') ?
    `logical_scalar_size.M ${logical_scalar_size.M}` :
    `logical_scalar_size.K ${logical_scalar_size.K}`}`
  );
  assert(
    input_A.buffer_layout.buffer_outer_boundary * input_A.buffer_layout.packed_vector_size ===
        (input_A.buffer_layout.loading_points_layout === 'NHW' ? logical_scalar_size.K : logical_scalar_size.M),
    `Expect for layout ${input_A.buffer_layout.loading_points_layout} \
input_A.buffer_layout.buffer_outer_boundary ${input_A.buffer_layout.buffer_outer_boundary} * \
    input_A.buffer_layout.packed_vector_size ${input_A.buffer_layout.packed_vector_size} === \
${(input_A.buffer_layout.loading_points_layout === 'NHW') ?
    `logical_scalar_size.K ${logical_scalar_size.K}` :
    `logical_scalar_size.M ${logical_scalar_size.M}`}`
  );
  const input_A_tensor_loading_points_shape = [
    input_A.aggregated_batches,
    // input_A.buffer_layout.loading_points_layout === 'NHW' ? logical_scalar_size.M : logical_scalar_size.K,
    // input_A.buffer_layout.loading_points_layout === 'NHW' ? logical_scalar_size.K : logical_scalar_size.M,
    input_A.buffer_layout.buffer_inner_boundary,
    input_A.buffer_layout.buffer_outer_boundary,
  ];

  assert(
    input_B.buffer_layout.buffer_inner_boundary ===
        ((input_B.buffer_layout.loading_points_layout === 'NHW') ? logical_scalar_size.K : logical_scalar_size.N),
    `Expect for layout ${input_B.buffer_layout.loading_points_layout} \
input_B.buffer_layout.buffer_inner_boundary ${input_B.buffer_layout.buffer_inner_boundary} === \
${(input_B.buffer_layout.loading_points_layout === 'NHW') ?
    `logical_scalar_size.K ${logical_scalar_size.K}` :
    `logical_scalar_size.N ${logical_scalar_size.N}`}`
  );
  assert(
    input_B.buffer_layout.buffer_outer_boundary * input_B.buffer_layout.packed_vector_size ===
        (input_B.buffer_layout.loading_points_layout === 'NHW' ? logical_scalar_size.N : logical_scalar_size.K),
    `Expect for layout ${input_B.buffer_layout.loading_points_layout} \
input_B.buffer_layout.buffer_outer_boundary ${input_B.buffer_layout.buffer_outer_boundary} * \
    input_B.buffer_layout.packed_vector_size ${input_B.buffer_layout.packed_vector_size} === \
${(input_B.buffer_layout.loading_points_layout === 'NHW') ?
    `logical_scalar_size.N ${logical_scalar_size.N}` :
    `logical_scalar_size.K ${logical_scalar_size.K}`}`
  );
  const input_B_tensor_loading_points_shape = [
    input_B.aggregated_batches,
    // input_B.buffer_layout.loading_points_layout === 'NHW' ? logical_scalar_size.K : logical_scalar_size.N,
    // input_B.buffer_layout.loading_points_layout === 'NHW' ? logical_scalar_size.N : logical_scalar_size.K,
    input_B.buffer_layout.buffer_inner_boundary,
    input_B.buffer_layout.buffer_outer_boundary,
  ];
  assert(output.buffer_layout.loading_points_layout === 'NHW', `Currently output tensor must be in NHW layout`)
  assert(
    output.buffer_layout.buffer_inner_boundary ===
        ((output.buffer_layout.loading_points_layout === 'NHW') ? logical_scalar_size.M : logical_scalar_size.N),
    `Expect for layout ${output.buffer_layout.loading_points_layout} \
output.buffer_layout.buffer_inner_boundary ${output.buffer_layout.buffer_inner_boundary} === \
${(output.buffer_layout.loading_points_layout === 'NHW') ?
    `logical_scalar_size.M ${logical_scalar_size.M}` :
    `logical_scalar_size.N ${logical_scalar_size.N}`}`
  );
  assert(
    output.buffer_layout.buffer_outer_boundary * output.buffer_layout.packed_vector_size ===
        (output.buffer_layout.loading_points_layout === 'NHW' ? logical_scalar_size.N : logical_scalar_size.M),
    `Expect for layout ${output.buffer_layout.loading_points_layout} \
output.buffer_layout.buffer_outer_boundary ${output.buffer_layout.buffer_outer_boundary} * \
    output.buffer_layout.packed_vector_size ${output.buffer_layout.packed_vector_size} === \
${(output.buffer_layout.loading_points_layout === 'NHW') ?
    `logical_scalar_size.N ${logical_scalar_size.N}` :
    `logical_scalar_size.M ${logical_scalar_size.M}`}`
  );
  const output_tensor_loading_points_shape = [
    output.aggregated_batches,
    // output.buffer_layout.loading_points_layout === 'NHW' ? logical_scalar_size.M : logical_scalar_size.N,
    // output.buffer_layout.loading_points_layout === 'NHW' ? logical_scalar_size.N : logical_scalar_size.M,
    output.buffer_layout.buffer_inner_boundary,
    output.buffer_layout.buffer_outer_boundary,
  ];
  const unaggregated_output_tensor_shape = [...batches_info.output_batch_dims, output.buffer_layout.buffer_inner_boundary, output.buffer_layout.buffer_outer_boundary * output.buffer_layout.packed_vector_size];
  const input_tensors_shape = [input_A_tensor_loading_points_shape, input_B_tensor_loading_points_shape, ...(has_bias ? [output_tensor_loading_points_shape] : [])];
  programUniforms.push(...createTensorShapeVariables(...input_tensors_shape));
  programUniforms.push(...createTensorShapeVariables(output_tensor_loading_points_shape));

  const trival_shader =
      (shader_helper: ShaderHelper) => {
        // -----------------------------------------------------------------------------
        //   Input/output variables
        // -----------------------------------------------------------------------------
        const input_A_variable =
            inputVariable('input_A', tensor_data_type, 3, input_A.buffer_layout.packed_vector_size);
        const input_B_variable =
            inputVariable('input_B', tensor_data_type, 3, input_B.buffer_layout.packed_vector_size);
        const input_Bias_variable = has_bias ?
            inputVariable('input_Bias', tensor_data_type, 3, input_Bias.buffer_layout.packed_vector_size) :
            undefined;
        const output_variable = outputVariable('output', tensor_data_type, 3, output.buffer_layout.packed_vector_size);

        const input_variables = [input_A_variable, input_B_variable, ...(has_bias ? [input_Bias_variable!] : [])];
        // -----------------------------------------------------------------------------
        //   WGSL types
        // -----------------------------------------------------------------------------
        assert(
            input_A.type === input_B.type && (!has_bias || input_B.type === input_Bias.type),
            `Expect types of input_A ${input_A.type} and input_B ${input_B.type} ${
                has_bias ? `and input_Bias ${input_Bias.type} ` : ''}are the same`);
        ;
        const scalar_type_WGSL = input_A.type;
        const scalar_or_vector_type_WGSL = (packed_size: number) =>
            packed_size > 1 ? `vec${packed_size}<${scalar_type_WGSL}>` : scalar_type_WGSL
        const input_A_packed_type_WGSL = scalar_or_vector_type_WGSL(input_A.buffer_layout.packed_vector_size);
        const input_B_packed_type_WGSL = scalar_or_vector_type_WGSL(input_B.buffer_layout.packed_vector_size);
        // const output_packed_vector_size = maximumPackingSize(packing_shape_B.cols);
        assert(output.buffer_layout.packed_vector_direction === 'row', `Expect output buffer packing along row`);
        const output_packed_type_WGSL = scalar_or_vector_type_WGSL(output.buffer_layout.packed_vector_size);
        const compute_block_output_type_WGSL = `array<array<${output_packed_type_WGSL}, ${
            loading_points_per_compute_block_Output.cols}>, ${loading_points_per_compute_block_Output.rows}>`;

        // -----------------------------------------------------------------------------
        //   Helper functions, expressions, and statements
        // -----------------------------------------------------------------------------
        const helper_functions: Map<string /* function name */, string /* function definition */> =
            new Map<string, string>;

        helper_functions.set(
            'ConvertOutputBatchToInputBatch',
            `
// Convert output aggregated batch to vec2(input_A_aggregated_batch, input_B_aggregated_batch)
fn ConvertOutputBatchToInputBatch(output_batch: u32) -> vec2<u32> {
    var input_A_batch: u32 = 0;
    var input_B_batch: u32 = 0;
    var degregating_output_batch = output_batch;
    ${
                arrayMap(
                    Math.max(input_A_batch_dims.length, input_B_batch_dims.length),
                    i => `
    {
        let dim = degregating_output_batch % ${output_batch_dims[output_batch_dims.length - 1 - i]};
        degregating_output_batch = degregating_output_batch / ${output_batch_dims[output_batch_dims.length - 1 - i]};
        ${
                        ((i < input_A_batch_dims.length) &&
                         (!input_A_broadcasted_dims.includes(input_A_batch_dims.length - 1 - i))) ?
                            `
        input_A_batch += dim * ${input_A_batch_dims.slice(input_A_batch_dims.length - i).reduce((a, b) => a * b, 1)};
` :
                            ''}
        ${
                        ((i < input_B_batch_dims.length) &&
                         (!input_B_broadcasted_dims.includes(input_B_batch_dims.length - 1 - i))) ?
                            `
        input_B_batch += dim * ${input_B_batch_dims.slice(input_B_batch_dims.length - i).reduce((a, b) => a * b, 1)};
` :
                            ''}
    }`).join('\n')}

    return vec2<u32>(input_A_batch, input_B_batch);
}`);

        const createLoadBufferHelperFunction = (
            function_name: string,
            buffer_variable: IndicesHelper,
            packed_type_in_buffer_WGSL: string,
            buffer_layout: BufferLayoutInfo,
            ) => {
          if (buffer_variable.rank !== 3) {
            throw new Error(`Expected buffer A of rank 3, got ${buffer_variable.rank}.`);
          }
          assert(!helper_functions.has(function_name), `Redefining function ${function_name}`);

          const {loading_points_layout, buffer_inner_boundary, buffer_outer_boundary} = buffer_layout;

          helper_functions.set(function_name, `
// Load a loading point at logical position [batch][row][col] from a matrix buffer.
fn ${function_name}(loading_point_row: i32, loading_point_col: i32, batch: i32) -> ${packed_type_in_buffer_WGSL} {
    var value = ${packed_type_in_buffer_WGSL}(0.0);

    // Handle possible transposed layout
    let inner_dim = ${loading_points_layout === 'NHW' ? 'loading_point_row' : 'loading_point_col'};
    let outer_dim = ${loading_points_layout === 'NHW' ? 'loading_point_col' : 'loading_point_row'};

    if (inner_dim >= 0 && inner_dim < ${buffer_inner_boundary} && outer_dim >= 0 && outer_dim < ${buffer_outer_boundary})
    {
        // Within boundary, read from buffer[batch][inner_dim][outer_dim].
        var indices: ${buffer_variable.type.indices};
        indices[0] = u32(batch);
        indices[1] = u32(inner_dim);
        indices[2] = u32(outer_dim);
        value = ${buffer_variable.getByIndices('indices')};
    }
    return value;
}`);
          return function_name;
        };

        const buffer_A_loading_function_name = createLoadBufferHelperFunction(
            'LoadFromBufferA', input_A_variable, input_A_packed_type_WGSL, input_A.buffer_layout);
        const buffer_B_loading_function_name = createLoadBufferHelperFunction(
            'LoadFromBufferB', input_B_variable, input_B_packed_type_WGSL, input_B.buffer_layout);
        const buffer_Bias_loading_function_name = has_bias ?
            createLoadBufferHelperFunction(
                'LoadFromBufferB', input_Bias_variable!, output_packed_type_WGSL, input_Bias.buffer_layout) :
            undefined;

        const createStoreBufferHelperFunction = (
            function_name: string,
            buffer_variable: IndicesHelper,
            packed_type_in_buffer_WGSL: string,
            buffer_layout: BufferLayoutInfo,
            ) => {
          if (buffer_variable.rank !== 3) {
            throw new Error(`Expected buffer A of rank 3, got ${buffer_variable.rank}.`);
          }
          assert(!helper_functions.has(function_name), `Redefining function ${function_name}`);

          const {loading_points_layout, buffer_inner_boundary, buffer_outer_boundary} = buffer_layout;

            helper_functions.set(function_name, `
// Store a loading point at logical position [batch][row][col] from a matrix buffer.
fn ${function_name}(value: ${packed_type_in_buffer_WGSL}, loading_point_row: i32, loading_point_col: i32, batch: i32) {
    // Handle possible transposed layout
    let inner_dim = ${loading_points_layout === 'NHW' ? 'loading_point_row' : 'loading_point_col'};
    let outer_dim = ${loading_points_layout === 'NHW' ? 'loading_point_col' : 'loading_point_row'};

    if (inner_dim >= 0 && inner_dim < ${buffer_inner_boundary} && outer_dim >= 0 && outer_dim < ${buffer_outer_boundary})
    {
        // Within boundary, read from buffer[batch][inner_dim][outer_dim].
        var indices: ${buffer_variable.type.indices};
        indices[0] = u32(batch);
        indices[1] = u32(inner_dim);
        indices[2] = u32(outer_dim);
        ${buffer_variable.setByIndices('indices', 'value')};
    }
}`);
          return function_name;
        };

        const buffer_Output_storing_function_name = createStoreBufferHelperFunction(
            'StoreToBufferOutput', output_variable, output_packed_type_WGSL, output.buffer_layout);

        const callBufferLoadingFunctionExprBuilder = (buffer_loading_function: string, batch_variable: string) =>
            ((loading_point_row: number|string, loading_point_col: number|string) =>
                 `${buffer_loading_function}(${loading_point_row}, ${loading_point_col}, ${batch_variable})`);
        const callBufferALoadingFunctionExpr =
            callBufferLoadingFunctionExprBuilder(buffer_A_loading_function_name, 'batch_A');
        const callBufferBLoadingFunctionExpr =
            callBufferLoadingFunctionExprBuilder(buffer_B_loading_function_name, 'batch_B');
        const callBufferBiasLoadingFunctionExpr = has_bias ?
            callBufferLoadingFunctionExprBuilder(buffer_Bias_loading_function_name!, 'batch_output') :
            undefined;

        const loadInputBlockFromBufferStatsBuilder = (
            loading_points_per_compute_block: RowsColsSpan,
            callBufferLoadingFunctionExpr: (loading_point_row: number|string, loading_point_col: number|string) =>
                string,
            ) =>
            ((
                 compute_block_position_row: number|string,
                 compute_block_position_col: number|string,
                 loading_target_WGSL: string|string[][],
                 ident: number = 0,
                 ): string => {
              const loading_point_row_base = (typeof compute_block_position_row === 'number') ?
                  compute_block_position_row * loading_points_per_compute_block.rows :
                  `(${compute_block_position_row} * ${loading_points_per_compute_block.rows})`;
              const loading_point_col_base = (typeof compute_block_position_col === 'number') ?
                  compute_block_position_col * loading_points_per_compute_block.cols :
                  `(${compute_block_position_col} * ${loading_points_per_compute_block.cols})`;
              return addIdent(
                  arrayMap(
                      loading_points_per_compute_block.rows,
                      (row) =>
                          arrayMap(
                              loading_points_per_compute_block.cols,
                              (col) => `${
                                  typeof loading_target_WGSL === 'string' ? `${loading_target_WGSL}[${row}][${col}]` :
                                                                            loading_target_WGSL[row][col]} = \
${
                                  callBufferLoadingFunctionExpr(
                                      `i32(${loading_point_row_base} + ${row})`,
                                      `i32(${loading_point_col_base} + ${col})`)};`)
                              .join('\n'))
                      .join('\n'),
                  ident, 'keepFirstLine');
            });
        const loadInputABlockFromBufferStats =
            loadInputBlockFromBufferStatsBuilder(loading_points_per_compute_block_A, callBufferALoadingFunctionExpr);
        const loadInputBBlockFromBufferStats =
            loadInputBlockFromBufferStatsBuilder(loading_points_per_compute_block_B, callBufferBLoadingFunctionExpr);

        // Using vectorized computation statements to compute a single compute block
        const calcComputeBlockStats = (
            inputAVectorWGSLs: string[/* row groups */][/* col groups */],
            inputBVectorWGSLs: string[/* row groups */][/* col groups */],
            outputPackedWGSLName: string /* Must be a 2d array of vec/sclr type */|string[][],
            outputSpanUpperLeftPositionPackedRow: string|number,
            outputSpanUpperLeftPositionPackedCol: string|number,
            computeSchema: 'dotProduct'|'scaledVector',
            ) => {
          // Divide compute block into vectorized compute statements
          const vectorized_computations_shape: MKNSpan = {
            M: packing_shape_A.rows,
            K: Math.max(packing_shape_A.cols, packing_shape_B.rows),
            N: packing_shape_Output.cols,
          };
          const input_A_groups_per_vectorized_computation: RowsColsSpan = {
            rows: vectorized_computations_shape.M / packing_shape_A.rows,
            cols: vectorized_computations_shape.K / packing_shape_A.cols,
          };
          const input_B_groups_per_vectorized_computation: RowsColsSpan = {
            rows: vectorized_computations_shape.K / packing_shape_B.rows,
            cols: vectorized_computations_shape.N / packing_shape_B.cols,
          };
          const output_groups_per_vectorized_computation: RowsColsSpan = {
            rows: vectorized_computations_shape.M / packing_shape_Output.rows,
            cols: vectorized_computations_shape.N / packing_shape_Output.cols,
          };
          assert(
              MKNDivisible(compute_block_shape, vectorized_computations_shape),
              `Expect compute_block_shape ${
                  PrintMKN(compute_block_shape)} can be divided by vectorized_computations_shape ${
                  vectorized_computations_shape}`);
          const vectorized_computations_per_compute_block =
              MKNDivision(compute_block_shape, vectorized_computations_shape);
          return (
              arrayMap(
                  vectorized_computations_per_compute_block.M,
                  (vectorized_computation_M) =>
                      arrayMap(
                          vectorized_computations_per_compute_block.N,
                          (vectorized_computation_N) =>
                              arrayMap(
                                  vectorized_computations_per_compute_block.K,
                                  (vectorized_computation_K) => generateVectorizedComputeStatements(
                                      inputAVectorWGSLs
                                          .slice(
                                              vectorized_computation_M * input_A_groups_per_vectorized_computation.rows,
                                              (vectorized_computation_M + 1) *
                                                  input_A_groups_per_vectorized_computation.rows)
                                          .map(
                                              col_groups => col_groups.slice(
                                                  vectorized_computation_K *
                                                      input_A_groups_per_vectorized_computation.cols,
                                                  (vectorized_computation_K + 1) *
                                                      input_A_groups_per_vectorized_computation.cols)),
                                      inputBVectorWGSLs
                                          .slice(
                                              vectorized_computation_K * input_B_groups_per_vectorized_computation.rows,
                                              (vectorized_computation_K + 1) *
                                                  input_B_groups_per_vectorized_computation.rows)
                                          .map(
                                              col_groups => col_groups.slice(
                                                  vectorized_computation_N *
                                                      input_B_groups_per_vectorized_computation.cols,
                                                  (vectorized_computation_N + 1) *
                                                      input_B_groups_per_vectorized_computation.cols)),
                                      outputPackedWGSLName,
                                      (typeof outputSpanUpperLeftPositionPackedRow === 'string') ?
                                          `(${outputSpanUpperLeftPositionPackedRow} + ${
                                              vectorized_computation_M *
                                              output_groups_per_vectorized_computation.rows})` :
                                          outputSpanUpperLeftPositionPackedRow +
                                              vectorized_computation_M * output_groups_per_vectorized_computation.rows,
                                      (typeof outputSpanUpperLeftPositionPackedCol === 'string') ?
                                          `(${outputSpanUpperLeftPositionPackedCol} + ${
                                              vectorized_computation_N *
                                              output_groups_per_vectorized_computation.cols})` :
                                          outputSpanUpperLeftPositionPackedCol +
                                              vectorized_computation_N * output_groups_per_vectorized_computation.cols,
                                      scalar_type_WGSL, {
                                        inputAVectorSize: input_A.buffer_layout.packed_vector_size,
                                        inputAVectorDirection: input_A.buffer_layout.packed_vector_direction,
                                        inputBVectorSize: input_B.buffer_layout.packed_vector_size,
                                        inputBVectorDirection: input_B.buffer_layout.packed_vector_direction,
                                        outputVectorSize: output.buffer_layout.packed_vector_size,
                                        computeSchema,
                                        // Assume output vector is row vector.
                                        // Assume outputColGroups = 1
                                        expectedOutputRowGroups: vectorized_computations_shape.M as (1 | 2 | 4),
                                      }))
                                  .join('\n'))
                          .join('\n'))
                  .join('\n'));
        };

        // target_compute_block += src_compute_block
        const addOutputBlocksStats = (target_block_WGSL: string, src_block_WGSL: string) =>
            arrayMap(
                loading_points_per_compute_block_Output.rows,
                (row) => arrayMap(
                             loading_points_per_compute_block_Output.cols,
                             (col) => `${target_block_WGSL}[${row}][${col}] += ${src_block_WGSL}[${row}][${col}];`)
                             .join('\n'))
                .join('\n');

        const writebackOutputBlockWithBiasAndActivationStats = (
            acc_array_WGSL: string,
            compute_blocks_per_thread_M: string|number,
            compute_blocks_per_thread_N: string|number,
            ): string => {
          return `
let output_loading_point_thread_global_base_row = (compute_blocks_thread_workgroup_base_M + compute_blocks_workgroup_global_base_M) * ${
              loading_points_per_compute_block_Output.rows};
let output_loading_point_thread_global_base_col = (compute_blocks_thread_workgroup_base_N + compute_blocks_workgroup_global_base_N) * ${
              loading_points_per_compute_block_Output.cols};
var output_loading_point_block_global_base_row = output_loading_point_thread_global_base_row;
${
                u32LoopUpFrom0WGSL(
                    'compute_block_thread_M', compute_blocks_per_thread_M,
                    (compute_block_thread_M: string|number) => `
    // compute_block_thread_M loop
    var output_loading_point_block_global_base_col = output_loading_point_thread_global_base_col;
    ${
                        u32LoopUpFrom0WGSL(
                            'compute_block_thread_N', compute_blocks_per_thread_N,
                            (compute_block_thread_N: string|number) => `
        // compute_block_thread_N loop
        let output_block = &${acc_array_WGSL}[${compute_block_thread_M}][${compute_block_thread_N}];
        ${
                                arrayMap(
                                    loading_points_per_compute_block_Output.rows,
                                    (row) =>  //
                                    arrayMap(
                                        loading_points_per_compute_block_Output.cols,
                                        (col) => `
        {
            let output_vector = &((*output_block)[${row}][${col}]);
            let output_loading_points_global_row = output_loading_point_block_global_base_row + ${row};
            let output_loading_points_global_col = output_loading_point_block_global_base_col + ${col};
            // Add bias if any
            ${
                                            has_bias ? `
            *output_vector += ${
                                                           callBufferBiasLoadingFunctionExpr!
                                                           ('output_loading_points_global_row',
                                                            'output_loading_points_global_col')};` :
                                                       ''}
            // Handle activation if any
            ${getActivationSnippet(activation, output_packed_type_WGSL, scalar_type_WGSL, '(*output_vector)')}
            // Write back output
            ${buffer_Output_storing_function_name}(*output_vector, i32(output_loading_points_global_row), i32(output_loading_points_global_col), i32(batch_output));
        }`).join('\n')).join('\n')}
        output_loading_point_block_global_base_col += ${loading_points_per_compute_block_Output.cols};`,
                            /* outer_ident_width */ 4)}
    output_loading_point_block_global_base_row += ${loading_points_per_compute_block_Output.rows};`)}
`;
        };

        // -----------------------------------------------------------------------------
        //   Shader code template
        // -----------------------------------------------------------------------------
        const shader = `
// Matmul templated shader

const WGS = ${workgroup_params.threads_per_workgroup}u;
const tensor_slice_groups = ${tensor_slice_factor}u;
const threads_per_TSG = ${threads_per_TSG}u;  // WGS / tensor_slice_groups
const_assert(tensor_slice_groups * threads_per_TSG == WGS);

const compute_block_shape_M = ${compute_block_shape.M}u;
const compute_block_shape_K = ${compute_block_shape.K}u;
const compute_block_shape_N = ${compute_block_shape.N}u;

const compute_blocks_per_workgroup_M = ${compute_blocks_per_workgroup.M}u;
const compute_blocks_per_workgroup_K = ${compute_blocks_per_workgroup.K}u;
const compute_blocks_per_workgroup_N = ${compute_blocks_per_workgroup.N}u;

const threads_along_M_per_TSG = ${threads_along_M_per_TSG}u;
const threads_along_N_per_TSG = ${threads_along_N_per_TSG}u;
const_assert(threads_along_M_per_TSG * threads_along_N_per_TSG == threads_per_TSG);

const compute_blocks_per_thread_K = ${Math.ceil(compute_blocks_per_workgroup.K / tensor_slice_factor)}u;
const compute_blocks_per_thread_M = ${
            compute_blocks_per_thread_M}u;  // compute_blocks_per_workgroup_M / threads_along_M_per_TSG
const compute_blocks_per_thread_N = ${
            compute_blocks_per_thread_N}u;  // compute_blocks_per_workgroup_N / threads_along_N_per_TSG
const_assert(threads_along_M_per_TSG * compute_blocks_per_thread_M == compute_blocks_per_workgroup_M);
const_assert(threads_along_N_per_TSG * compute_blocks_per_thread_N == compute_blocks_per_workgroup_N);

alias InputAPackedType = ${input_A_packed_type_WGSL};
alias InputBPackedType = ${input_B_packed_type_WGSL};
alias OutputPackedType = ${output_packed_type_WGSL};    // Bias, if any, should have the same packed type.
// 2D array type holding a whole output block
alias ComputeBlockOutputType = ${compute_block_output_type_WGSL};

${shader_helper.registerUniforms(uniforms_WGSL_info).declareVariables(...input_variables, output_variable)}

${
            tensor_slice_factor > 1 ? `
// Shared memory for adding up tensor sliced results
var<workgroup> tensor_slice_acc: array<array<array<ComputeBlockOutputType, compute_blocks_per_workgroup_N>, compute_blocks_per_workgroup_M>, ${
                                          Math.ceil(tensor_slice_factor / 2)}>;
` :
                                      ''}

// Helper functions
${[...helper_functions.values()].join('\n\n')}

@compute @workgroup_size(WGS, 1, 1)
fn main(
    @builtin(workgroup_id) wid: vec3u,
    @builtin(local_invocation_id) lid: vec3u,
) {
    let compute_blocks_workgroup_global_base_M = wid.x * compute_blocks_per_workgroup_M;
    // compute_blocks_workgroup_global_base_K == 0
    let compute_blocks_workgroup_global_base_N = wid.y * compute_blocks_per_workgroup_N;
    let batch_output = wid.z;
    let input_batches = ConvertOutputBatchToInputBatch(batch_output);
    let batch_A = i32(input_batches.x);
    let batch_B = i32(input_batches.y);

    let thread_id_in_workgroup = lid.x;
    let tensor_slice_group = ${tensor_slice_factor === 1 ? '0u' : 'thread_id_in_workgroup / threads_per_TSG'};
    let thread_id_in_TSG = ${
            tensor_slice_factor === 1 ? 'thread_id_in_workgroup' : 'thread_id_in_workgroup % threads_per_TSG'};
    // Threads arranged in M major manner within a TSG, could also be N major.
    const threads_within_TSG_major_M_not_N = true;
    let thread_M_in_TSG = select(thread_id_in_TSG / threads_along_N_per_TSG, thread_id_in_TSG % threads_along_M_per_TSG, threads_within_TSG_major_M_not_N);
    let thread_N_in_TSG = select(thread_id_in_TSG % threads_along_N_per_TSG, thread_id_in_TSG / threads_along_M_per_TSG, threads_within_TSG_major_M_not_N);

    let compute_blocks_thread_workgroup_base_M = thread_M_in_TSG * compute_blocks_per_thread_M;
    let compute_blocks_thread_workgroup_base_K = tensor_slice_group * compute_blocks_per_thread_K;
    let compute_blocks_thread_workgroup_base_N = thread_N_in_TSG * compute_blocks_per_thread_N;

    var acc_array: array<array<ComputeBlockOutputType, compute_blocks_per_thread_N>, compute_blocks_per_thread_M>;

    const compute_block_thread_K_inner_loop_step: u32 = ${compute_block_thread_K_inner_loop_step};

    for (
        var compute_block_thread_K_outer_loop: u32 = 0;
        compute_block_thread_K_outer_loop < compute_blocks_per_thread_K;
        compute_block_thread_K_outer_loop += compute_block_thread_K_inner_loop_step
    ) {
        // Cache steps along K should be handled here.
        for (var compute_block_thread_M: u32 = 0; compute_block_thread_M < compute_blocks_per_thread_M; compute_block_thread_M++) {
            for (var compute_block_thread_N: u32 = 0; compute_block_thread_N < compute_blocks_per_thread_N; compute_block_thread_N++) {
                let acc: ptr<function, ComputeBlockOutputType> = &acc_array[compute_block_thread_M][compute_block_thread_N];
                let compute_block_M_workgroup_biased = compute_block_thread_M + compute_blocks_thread_workgroup_base_M;
                let compute_block_N_workgroup_biased = compute_block_thread_N + compute_blocks_thread_workgroup_base_N;
                let compute_block_M_global_biased = compute_block_thread_M + compute_blocks_thread_workgroup_base_M + compute_blocks_workgroup_global_base_M;
                let compute_block_N_global_biased = compute_block_thread_N + compute_blocks_thread_workgroup_base_N + compute_blocks_workgroup_global_base_N;

                ${
            u32LoopUpFrom0WGSL(
                'compute_block_thread_K_inner_loop', compute_block_thread_K_inner_loop_step,
                (compute_block_thread_K_inner_loop_var_or_value) => `
                    let compute_block_thread_K = compute_block_thread_K_outer_loop + ${
                    compute_block_thread_K_inner_loop_var_or_value};
                    // compute_block_K is unbiased from workgroup to global
                    let compute_block_K_global_biased = compute_block_thread_K + compute_blocks_thread_workgroup_base_K;
                    /*
                    Handle compute block:
                        let input_A_block = loadABlock(compute_block_M_biased, compute_block_K_biased, compute_block_N_biased);
                        let input_B_block = loadBBlock(compute_block_M_biased, compute_block_K_biased, compute_block_N_biased);
                        computeBlock(acc, input_A_block, input_B_block);
                    */
                    // Load A input block from LLC/memory
                    var compute_block_input_A: array<array<${input_A_packed_type_WGSL}, ${
                    loading_points_per_compute_block_A.cols}>, ${loading_points_per_compute_block_A.rows}>;
                    ${
                    loadInputABlockFromBufferStats(
                        'compute_block_M_global_biased', 'compute_block_K_global_biased', 'compute_block_input_A', 20)}
                    // Load B input block from LLC/memory
                    var compute_block_input_B: array<array<${input_B_packed_type_WGSL}, ${
                    loading_points_per_compute_block_B.cols}>, ${loading_points_per_compute_block_B.rows}>;
                    ${
                    loadInputBBlockFromBufferStats(
                        'compute_block_K_global_biased', 'compute_block_N_global_biased', 'compute_block_input_B', 20)}
                    // Compute block
                    ${
                    calcComputeBlockStats(
                        Array.from({length: loading_points_per_compute_block_A.rows})
                            .map(
                                (_, row) => Array.from({length: loading_points_per_compute_block_A.cols})
                                                .map((_, col) => `compute_block_input_A[${row}][${col}]`)),
                        Array.from({length: loading_points_per_compute_block_B.rows})
                            .map(
                                (_, row) => Array.from({length: loading_points_per_compute_block_B.cols})
                                                .map((_, col) => `compute_block_input_B[${row}][${col}]`)),
                        '(*acc)', 0, 0, compute_schema)}
`)}
            }
        }
    }


    ${
            tensor_slice_factor > 1 ? `
    // Add up tensor slice results if necessary
    ${(() => {
              let code = '';
              for (let remaining_slices = tensor_slice_factor; remaining_slices > 1;
                   remaining_slices = Math.ceil(remaining_slices / 2)) {
                code += `
    // Merge tensor slices ${remaining_slices} -> ${Math.ceil(remaining_slices / 2)}
    // Sub-step A: Store upper-half reg to lower-half SM
    if (tensor_slice_group >= ${Math.ceil(remaining_slices / 2)}) {
        for (var compute_block_thread_M: u32 = 0; compute_block_thread_M < compute_blocks_per_thread_M; compute_block_thread_M++) {
            for (var compute_block_thread_N: u32 = 0; compute_block_thread_N < compute_blocks_per_thread_N; compute_block_thread_N++) {
                let reg_acc = &acc_array[compute_block_thread_M][compute_block_thread_N];
                tensor_slice_acc
                    [tensor_slice_group-${Math.ceil(remaining_slices / 2)}]
                    [compute_blocks_thread_workgroup_base_M+compute_block_thread_M]
                    [compute_blocks_thread_workgroup_base_N+compute_block_thread_N] =
                    *reg_acc;
            }
        }
    }
    workgroupBarrier();
    // Sub-step B: Lower-half add SM into reg
    if (tensor_slice_group < ${Math.ceil(remaining_slices / 2)}) {
        for (var compute_block_thread_M: u32 = 0; compute_block_thread_M < compute_blocks_per_thread_M; compute_block_thread_M++) {
            for (var compute_block_thread_N: u32 = 0; compute_block_thread_N < compute_blocks_per_thread_N; compute_block_thread_N++) {
                let workgroup_acc =
                      &tensor_slice_acc
                          [tensor_slice_group]
                          [compute_blocks_thread_workgroup_base_M+compute_block_thread_M]
                          [compute_blocks_thread_workgroup_base_N+compute_block_thread_N];
                let reg_acc = &acc_array[compute_block_thread_M][compute_block_thread_N];
                ${addIdent(addOutputBlocksStats('(*reg_acc)', '(*workgroup_acc)'), 16, 'keepFirstLine')}
            }
        }
    }
    workgroupBarrier();
`;
              }
              return code;
            })()}
` :
                                      ''}
    //Write back results in register
    ${(() => {
          const code: string = addIdent(
              writebackOutputBlockWithBiasAndActivationStats(
                  'acc_array', 'compute_blocks_per_thread_M', 'compute_blocks_per_thread_N'),
              4, 'addAllLines');
          return tensor_slice_factor > 1 ? `
    if (tensor_slice_group == 0) {
        ${addIdent(code, 4, 'addAllLines')}
    }` :
                                           code;
        })()}
}

`;
        // LOG('fatal', `shader: \n ${shader}`);
        return shader;
      }

  const cacheKey = {
    activation,
    workgroup_params,
    compute_block_shape,
    compute_blocks_per_workgroup,
    tensor_slice_factor,
    input_A_layoout: input_A.buffer_layout,
    input_B_layoout: input_B.buffer_layout,
    output_layoout: output.buffer_layout,
  };

  // LOG(`fatal`, `templatedMatMulProgram Return`);

  return {
    name: 'MatMulTemplatedTrival',
    shaderCache: {
      // hint: `${activation.activation};${Object.entries(workgroup_params).map((entry) => entry.join(':')).join(';')};`,
      hint: `${Object.entries(cacheKey).map((entry) => entry.map(x => JSON.stringify(x)).join(':')).join(';')};`,
      inputDependencies: has_bias ? ['rank', 'rank', 'rank'] : ['rank', 'rank']
    },
    getShaderSource: trival_shader,
    getRunData: () => ({
      outputs: [{dims: unaggregated_output_tensor_shape, dataType: tensor_data_type}],
      dispatchGroup: dispatch_workgroups,
      programUniforms
    }),
  };
}

export const templatedMatMulDriver = (
    inputs: readonly TensorView[],
    activationAttributes: InternalActivationAttributes,
    outputShape: readonly number[],
    // reshapedOutputShape?: readonly number[],
    // isChannelsLast = false /* only used for conv2dByMatMul*/
    ):
    ProgramInfo => {
      const getBatchDims = (dims: readonly number[]) => dims.slice(0, -2);
      const output_batch_dims = getBatchDims(outputShape);
      const input_A_batch_dims = getBatchDims(inputs[0].dims);
      const input_B_batch_dims = getBatchDims(inputs[1].dims);
      // dimsAggregateBatches always return a 3-dims array, where the first element (batches) might be 1
      const dimsAggregateBatches =
          (dims: readonly number[]) => [getBatchDims(dims).reduce((a, b) => a * b, 1), ...dims.slice(-2)];
      const tensorAggregateBatches = (tensor: TensorView) => tensor.reshape(dimsAggregateBatches(tensor.dims));
      const batchAggregatedInputs = inputs.map(tensorAggregateBatches);
      const batchAggregatedOutputShape = dimsAggregateBatches(outputShape);
      /*
      assert(
          batchAggregatedInputs[0].dims[0] * batchAggregatedInputs[1].dims[0] === batchAggregatedOutputShape[0],
          `MatMul output batches ${batchAggregatedOutputShape[0]} should be equal to \
input A batches ${batchAggregatedInputs[0].dims[0]} * input B batches ${batchAggregatedInputs[1].dims[0]}`);
      */

      const createMatrixTensorInfo = (
          aggregated_tensor_dims: readonly number[],
          tensor_data_type: number,
          loading_points_layout: 'NHW'|'NWH',
          // packed_vector_direction: 'row'|'col',
          ): TensorInfo => {
        assert(
            [DataType.float, DataType.float16].includes(tensor_data_type),
            `Unknown tensor dataType ${tensor_data_type}, expected float or float16`);
        assert(
            aggregated_tensor_dims.length === 3,
            `Expect 3 dims for batch aggregated tansor, got ${aggregated_tensor_dims.length}`);
        const packed_vector_size = getMaxComponents(aggregated_tensor_dims[2]);
        return {
          aggregated_batches: aggregated_tensor_dims[0],
          rows: loading_points_layout === 'NHW' ? aggregated_tensor_dims[1] : aggregated_tensor_dims[2],
          cols: loading_points_layout === 'NHW' ? aggregated_tensor_dims[2] : aggregated_tensor_dims[1],
          type: tensor_data_type === DataType.float ? 'f32' : 'f16',
          buffer_layout: {
            packed_vector_direction: loading_points_layout === 'NHW' ? 'row':'col',
            packed_vector_size,
            loading_points_layout,
            // Buffer accessing: buffer_variable[batch][inner][outer] of packed type
            buffer_inner_boundary: aggregated_tensor_dims[1],
            buffer_outer_boundary: aggregated_tensor_dims[2] / packed_vector_size,
          },
        };
      };

      const tensor_data_type = batchAggregatedInputs[0].dataType;
      const op_params: GEMMOperationParameters = {
        batches_info: {
          output_batch_dims,
          input_A_batch_dims,
          input_B_batch_dims,
        },
        tensor_data_type,
        input_A: createMatrixTensorInfo(batchAggregatedInputs[0].dims, tensor_data_type, 'NHW'),
        input_B: createMatrixTensorInfo(batchAggregatedInputs[1].dims, tensor_data_type, 'NHW'),
        input_Bias: batchAggregatedInputs.length >= 3 ?
            createMatrixTensorInfo(batchAggregatedInputs[2].dims, tensor_data_type, 'NHW') :
            undefined,
        output: createMatrixTensorInfo(batchAggregatedOutputShape, tensor_data_type, 'NHW'),
        activation: activationAttributes
      };

      const workgroup_params: GEMMWorkgroupScheduleParameters = {
        output_rows_per_workgroup: 64,
        output_cols_per_workgroup: 64,
        threads_per_workgroup: 128,
        /*
        limits: {
            maximium_register_size: 32,
            maximium_shared_memory_size: ,
        },
        */
      };

      const schedule_params: ShaderScheduleSchemaParameters = {
        compute_schema: 'scaledVector',
        // compute_schema: 'dotProduct',
        tensor_slice: {
          // tensor_slice_factor: 1,
          tensor_slice_factor: 4,
          tensor_slice_input_policy: 'continious',
        },
      };

      LOG(`fatal`, `Before templatedMatMulProgram`);

      return templatedMatMulProgram(
          op_params,
          workgroup_params,
          schedule_params,
      );
    }

export const createComputeBlockMatmulNaiveProgramInfo =
    (inputs: readonly TensorView[], activationAttributes: InternalActivationAttributes, outputShape: readonly number[],
     reshapedOutputShape?: readonly number[],
     isChannelsLast = false /* only used for conv2dByMatMul*/): ProgramInfo => {
      const aShape = inputs[0].dims;
      const bShape = inputs[1].dims;

      const M = aShape[aShape.length - 2];
      const N = bShape[bShape.length - 1];
      const K = aShape[aShape.length - 1];
      const components = getMaxComponents(N);
      const aComponents = getMaxComponents(K);
      const outputNumber = getMaxComponents(M);
      const outputSize = ShapeUtil.size(outputShape) / components / outputNumber;
      const hasBias = inputs.length > 2;
      const outerDims = reshapedOutputShape ? reshapedOutputShape.slice(0, -2) : outputShape.slice(0, -2);
      const batchSize = ShapeUtil.size(outerDims);
      const outputShapeInShader = [batchSize, M, N];

      const programUniforms: ProgramUniform[] = [
        {type: DataType.uint32, data: outputSize}, {type: DataType.uint32, data: M}, {type: DataType.uint32, data: N},
        {type: DataType.uint32, data: K}
      ];
      appendActivationUniformsData(activationAttributes, programUniforms);
      programUniforms.push(...createTensorShapeVariables(outerDims, aShape, bShape));
      if (hasBias) {
        programUniforms.push(...createTensorShapeVariables(inputs[2].dims));
      }
      programUniforms.push(...createTensorShapeVariables(outputShapeInShader));

      const getShaderSource = (shader_helper: ShaderHelper) => {
        const batchDims = internalVariable('batch_dims', inputs[0].dataType, outerDims.length);
        const a = inputVariable('a', inputs[0].dataType, aShape.length, aComponents);
        const b = inputVariable('b', inputs[1].dataType, bShape.length, components);
        const output = outputVariable('output', inputs[0].dataType, outputShapeInShader.length, components);
        const baseType = tensorTypeToWsglStorageType(output.type.tensor);
        const applyActivation = getActivationSnippet(activationAttributes, output.type.value, baseType);
        const inputVariables = [a, b];
        let processBias = '';
        if (hasBias) {
          const biasComponents = isChannelsLast ? components : 1;
          inputVariables.push(inputVariable('bias', inputs[2].dataType, inputs[2].dims.length, biasComponents));
          processBias = `${
              isChannelsLast ? `value += bias[col / ${biasComponents}];` :
                               `value += ${output.type.value}(bias[row + i]);`}`;
        }

        const outerDimsA = aShape.slice(0, -2);
        const outerDimsB = bShape.slice(0, -2);
        const broadCastADims = getBroadcastDims(outerDimsA, outerDims);
        const broadCastBDims = getBroadcastDims(outerDimsB, outerDims);
        const uniforms: UniformsArrayType = [
          {name: 'output_size', type: 'u32'}, {name: 'M', type: 'u32'}, {name: 'N', type: 'u32'},
          {name: 'K', type: 'u32'}
        ];
        appendActivationUniforms(activationAttributes, uniforms);

        const getIndices = (variable: IndicesHelper, broadCastDims: number[]) => {
          const rank = variable.rank;
          const name = variable.name;
          if (rank === 2) {
            return `var ${name}_indices = ${variable.type.indices}(0u, 0u);`;
          }
          const batchRank = batchDims.rank;
          let resStr = `var ${name}_indices: ${variable.type.indices};`;
          for (let i = rank - 2 - 1, j = batchRank - 1; i >= 0; i--, j--) {
            resStr += `\n${name}_indices[${i}] = ${batchRank > 1 ? `batch_indices[${j}]` : 'batch_indices'};`;
          }
          broadCastDims.forEach(i => {
            resStr += `\n${name}_indices[${i}] = 0;`;
          });
          resStr += `${name}_indices[${rank - 2}] = 0u;
                     ${name}_indices[${rank - 1}] = 0u;`;
          return resStr;
        };

        const calcResult = (): string => {
          let calcStr = `var a_data: ${a.type.value};`;
          for (let i = 0; i < aComponents; i++) {
            calcStr += `
              let b_data${i} = b[(b_offset + (k + ${i}) * uniforms.N + col) / ${components}];`;
          }
          for (let i = 0; i < outputNumber; i++) {
            calcStr += `a_data = a[(a_offset + (row + ${i}) * uniforms.K + k) / ${aComponents}];`;

            for (let j = 0; j < aComponents; j++) {
              calcStr += `
            values[${i}] = fma(${b.type.value}(a_data${aComponents === 1 ? '' : `[${j}]`}), b_data${j}, values[${
                  i}]);\n`;
            }
          }
          return calcStr;
        };

        return `
  ${
            shader_helper.registerUniforms(uniforms).registerInternalVariables(batchDims).declareVariables(
                ...inputVariables, output)}
  ${shader_helper.mainStart()}
    ${shader_helper.guardAgainstOutOfBoundsWorkgroupSizes('uniforms.output_size')}
    let col = (global_idx % (uniforms.N / ${components})) * ${components};
    var index1 = global_idx / (uniforms.N / ${components});
    let stride1 = uniforms.M / ${outputNumber};
    let row = (index1 % stride1) * ${outputNumber};
    let batch = index1 / stride1;

    ${outputShape.length === 2 ? '' : `let batch_indices = ${batchDims.offsetToIndices('batch')};`}
    ${getIndices(a, broadCastADims)}
    let a_offset = ${a.indicesToOffset('a_indices')};
    ${getIndices(b, broadCastBDims)}
    let b_offset = ${b.indicesToOffset('b_indices')};
    var values: array<${output.type.value}, ${outputNumber}>;
    for (var k: u32 = 0u; k < uniforms.K; k = k + ${aComponents}) {
      ${calcResult()}
    }
    for (var i = 0u; i < ${outputNumber}u; i++) {
      var value = values[i];
      ${processBias}
      ${applyActivation}
      let cur_indices = ${output.type.indices}(batch, row + i, col);
      let offset = ${output.indicesToOffset('cur_indices')};
      ${output.setByOffset(`offset / ${components}`, 'value')};
    }
  }
  `;
      };
      return {
        name: 'MatMulNaive',
        shaderCache: {
          hint: `${activationAttributes.activation};${components};${aComponents};${outputNumber};${isChannelsLast}`,
          inputDependencies: hasBias ? ['rank', 'rank', 'rank'] : ['rank', 'rank']
        },
        getRunData: () => ({
          outputs: [{dims: outputShape, dataType: inputs[0].dataType}],
          dispatchGroup: {x: Math.ceil(outputSize / 64 /* workgroup size */)},
          programUniforms
        }),
        getShaderSource
      };
    };
