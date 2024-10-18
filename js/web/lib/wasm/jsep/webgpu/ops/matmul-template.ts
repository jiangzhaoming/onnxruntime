// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {DataType} from '../../../wasm-common';
import {LOG} from '../../log';
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

type ScheduleSchemaSubgroupsParameters = {
  use_subgroups: false;
}|{
  use_subgroups: true;
  subgroups_shared_on: 'A'|'B';
  // Consider adaptive subgroup size
  expected_subgroup_size: number;
  thread_loading_point_register_mapping_to_subgroup_block: 'adjacent'|'interleaved';
};

// Tensor slice divides inputs along K dimension,
// makes A[MxK]*B[KxN] into Ai[MxKs]*Bi[KsxN], Ks = K / factor,
// i = 0 to factor-1.
type ScheduleSchemaTensorSliceParameters = {
  tensor_slice_factor: number;
  // How to divide the input A and B along K dimension
  // tensor_slice_input_policy: 'continious' | 'interleaved';
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
          .map(
              (line, i) =>
                  (line === '' || ((first_line_hint === 'keepFirstLine') && (i === 0))) ? line : (ident + line))
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
    disable_unfold: boolean = false,
    outer_ident_width: number = 0,
    ) =>
    integerLoopUpFrom0WGSL(
        loop_var_name, 'u32', disable_unfold ? `${upper_boundary}` : upper_boundary, loop_body, outer_ident_width);

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
  // const inputBScalarAccessor = inputScalarAccessorBuilder(inputBVectorWGSLs, inputBVectorSize);

  const inputAScalarPositionAccessor = (scalarRow: number, scalarCol: number) => (inputAVectorDirection === 'row') ?
      `${inputAScalarAccessor(scalarRow, Math.floor(scalarCol / inputAVectorSize), scalarCol % inputAVectorSize)}` :
      `${inputAScalarAccessor(Math.floor(scalarRow / inputAVectorSize), scalarCol, scalarRow % inputAVectorSize)}`;
  /*
  const inputBScalarPositionAccessor = (scalarRow: number, scalarCol: number) => (inputBVectorDirection === 'row') ?
  `${inputBScalarAccessor(scalarRow, Math.floor(scalarCol / inputBVectorSize), scalarCol % inputBVectorSize)}` :
  `${inputBScalarAccessor(Math.floor(scalarRow / inputBVectorSize), scalarCol, scalarRow % inputBVectorSize)}`;
  */

  interface InputAccessingTrace {
    loading_point_row: number;
    loading_point_col: number;
    loading_point_components: number[];
  }
  ;
  const inputScalarPositionAccessingTracesBuilder = (inputVectorDirection: 'row'|'col', inputVectorSize: 1|2|4) =>
      ((scalarRow: number, scalarCol: number) => (inputVectorDirection === 'row') ?
           {
             loading_point_row: scalarRow,
             loading_point_col: Math.floor(scalarCol / inputVectorSize),
             loading_point_components: [scalarCol % inputVectorSize]
           } as InputAccessingTrace :
           {
             loading_point_row: Math.floor(scalarRow / inputVectorSize),
             loading_point_col: scalarCol,
             loading_point_components: [scalarRow % inputVectorSize]
           } as InputAccessingTrace);
  const inputAScalarPositionAccessingTraces =
      inputScalarPositionAccessingTracesBuilder(inputAVectorDirection, inputAVectorSize);
  const inputBScalarPositionAccessingTraces =
      inputScalarPositionAccessingTracesBuilder(inputBVectorDirection, inputBVectorSize);

  const aggregateInputAccessingTraces = (traces: readonly InputAccessingTrace[]) => traces.reduceRight(
      (reduced, previous) => (previous.loading_point_row === reduced[0]?.loading_point_row &&
                              previous.loading_point_col === reduced[0]?.loading_point_col) ?
          [
            {
              loading_point_row: reduced[0].loading_point_row,
              loading_point_col: reduced[0].loading_point_col,
              loading_point_components: [previous.loading_point_components, ...reduced[0].loading_point_components],
            } as InputAccessingTrace,
            ...reduced.slice(1)
          ] :
          [previous, ...reduced],
      [] as InputAccessingTrace[]);
  const vectorFromInputAccessingTracesBuilder =
      (inputVectorWGSLs: readonly(readonly string[])[], inputVectorSize: 1|2|4) =>
          (traces: readonly InputAccessingTrace[]) =>
              traces
                  .map(
                      trace => inputVectorSize === 1 ?
                          `${
                              trace.loading_point_components.length > 1 ?
                                  `vec${trace.loading_point_components.length}` :
                                  ''}(${inputVectorWGSLs[trace.loading_point_row][trace.loading_point_col]})` :
                          `${inputVectorWGSLs[trace.loading_point_row][trace.loading_point_col]}.${
                              trace.loading_point_components.map(i => 'xyzw'[i]).join('')}`)
                  .join(', ');
  const vectorFromInputAAccessingTraces = vectorFromInputAccessingTracesBuilder(inputAVectorWGSLs, inputAVectorSize);
  const vectorFromInputBAccessingTraces = vectorFromInputAccessingTracesBuilder(inputBVectorWGSLs, inputBVectorSize);

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
    const gatherRowVectorFromInputA = (scalarRow: number, scalarCol: number, vectorSize: 1|2|4) => {
      if ((scalarRow >= inputAScalarRows) || (scalarCol + vectorSize > kDimensionSize)) {
        throw (`gatherRowVectorFromInputA OOB, requiring [${scalarRow}][${scalarCol}:+${
            vectorSize}] on input A with scalars [${inputAScalarRows}][${kDimensionSize}]`);
      }
      return `${vectorSize === 1 ? scalarTypeWGSL : `vec${vectorSize}<${scalarTypeWGSL}>`}(${
          vectorFromInputAAccessingTraces(aggregateInputAccessingTraces(
              arrayMap(vectorSize, (col) => inputAScalarPositionAccessingTraces(scalarRow, scalarCol + col))))})`;
    };
    const gatherColVectorFromInputB = (scalarRow: number, scalarCol: number, vectorSize: 1|2|4) => {
      if ((scalarRow + vectorSize > kDimensionSize) || (scalarCol > inputBScalarCols)) {
        throw (`gatherColVectorFromInputB OOB, requiring [${scalarRow}:+${vectorSize}][${
            scalarCol}] on input B with scalars [${kDimensionSize}][${inputBScalarCols}]`);
      }
      return `${vectorSize === 1 ? scalarTypeWGSL : `vec${vectorSize}<${scalarTypeWGSL}>`}(${
          vectorFromInputBAccessingTraces(aggregateInputAccessingTraces(
              arrayMap(vectorSize, (row) => inputBScalarPositionAccessingTraces(scalarRow + row, scalarCol))))})`;
    };
    // Divide kDimensionSize to kDimensionSizeOf4 + kDimensionSizeOf2 + kDimensionSizeOf1
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
            vectorSize}] on input B with scalars [${inputBScalarRows}][${inputBScalarCols}]`);
      }
      return `${vectorSize === 1 ? scalarTypeWGSL : `vec${vectorSize}<${scalarTypeWGSL}>`}(${
          vectorFromInputBAccessingTraces(aggregateInputAccessingTraces(
              arrayMap(vectorSize, (col) => inputBScalarPositionAccessingTraces(scalarRow, scalarCol + col))))})`;
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


// Assign loading points WGSL to a target
const assignLoadingPointsToTargetStats = (
    loading_target_WGSL: string|string[][],
    loading_points_WGSL: string[][],
    ident: number = 0,
    ): string => {
  return addIdent(
      loading_points_WGSL
          .map(
              (loading_points_in_row, row) =>  //
              loading_points_in_row
                  .map(
                      (loading_point_WGSL, col) => `${
                          typeof loading_target_WGSL === 'string' ?
                              `${loading_target_WGSL}[${row}][${col}]` :
                              loading_target_WGSL[row][col]} = ${loading_point_WGSL};`)
                  .join('\n'))
          .join('\n'),
      ident, 'keepFirstLine');
};

interface TSGBlockCacheHelper {
  cacheMemoryModuleDefinitionWGSL: () => string;
  cacheMemoryFunctionDefinitionWGSL: () => string;
  cacheMemoryUpdateStatsWGSL: (unfoldStepLoop?: boolean) => string;
  get loadingPointAccessingInCacheExpr():
      (row_in_cache: string|number, col_in_cache: string|number, TSG: string) => string;
}

interface TSGBlockCacheHelperConstructorParams {
  variable_name: string;
  threads_per_TSG: number;
  loading_points_per_compute_block: RowsColsSpan;
  cache_compute_blocks_per_TSG: RowsColsSpan;
  loading_point_WGSL_type: string;
  loading_point_row_cache_step_TSG_base_WGSL: string;
  loading_point_col_cache_step_TSG_base_WGSL: string;
  sourceLoadingPointTSGAccessingExpr: (TSG: string, row_in_TSG: string, col_in_TSG: string) => string;
  loading_points_prefer_major: 'row'|'col';
}

class TSGBlockCacheHelperBase {
  readonly variable_name: string;
  // tensor_slice_groups: number;
  readonly threads_per_TSG: number;
  readonly loading_points_per_compute_block: RowsColsSpan;
  //
  readonly cache_compute_blocks_per_TSG: RowsColsSpan;
  readonly loading_point_WGSL_type: string;
  // loading_points_stride_per_TSG: RowsColsSpan;
  // Cache step base for updating
  readonly loading_point_row_cache_step_TSG_base_WGSL: string;
  readonly loading_point_col_cache_step_TSG_base_WGSL: string;
  readonly sourceLoadingPointTSGAccessingExpr: (TSG: string, row_in_TSG: string, col_in_TSG: string) => string;
  // Threads in lock step prefer loading loading points from the same row/col
  readonly loading_points_prefer_major: 'row'|'col';

  // Computed
  loading_points_per_cache_step_TSG: RowsColsSpan;
  cache_name: string;

  constructor(params: TSGBlockCacheHelperConstructorParams) {
    this.variable_name = params.variable_name;
    this.threads_per_TSG = params.threads_per_TSG;
    this.loading_points_per_compute_block = params.loading_points_per_compute_block;
    this.cache_compute_blocks_per_TSG = params.cache_compute_blocks_per_TSG;
    this.loading_point_WGSL_type = params.loading_point_WGSL_type;
    // this.loading_points_stride_per_TSG = loading_points_stride_per_TSG;
    this.loading_point_row_cache_step_TSG_base_WGSL = params.loading_point_row_cache_step_TSG_base_WGSL;
    this.loading_point_col_cache_step_TSG_base_WGSL = params.loading_point_col_cache_step_TSG_base_WGSL;
    this.sourceLoadingPointTSGAccessingExpr = params.sourceLoadingPointTSGAccessingExpr;
    this.loading_points_prefer_major = params.loading_points_prefer_major;

    this.loading_points_per_cache_step_TSG = {
      rows: this.cache_compute_blocks_per_TSG.rows * this.loading_points_per_compute_block.rows,
      cols: this.cache_compute_blocks_per_TSG.cols * this.loading_points_per_compute_block.cols,
    };
    this.cache_name = `cache_${this.variable_name}`;
  }

  loadingPointIdToPosition(loading_point_id: number|string, major_direction: 'row'|'col', major_stride: number):
      {loading_point_row: number|string; loading_point_col: number | string} {
    let loading_point_mod_stride: number|string;
    let loading_point_div_stride: number|string;
    if (typeof loading_point_id === 'number') {
      loading_point_mod_stride = loading_point_id % major_stride;
      loading_point_div_stride = loading_point_id / major_stride;
    } else {
      loading_point_mod_stride =
          ((Number.isInteger(Math.log2(major_stride))) ? `((${loading_point_id}) & ${major_stride - 1})` :
                                                         `((${loading_point_id}) % ${major_stride})`);
      loading_point_div_stride =
          ((Number.isInteger(Math.log2(major_stride))) ? `((${loading_point_id}) >> ${Math.log2(major_stride)})` :
                                                         `((${loading_point_id}) / ${major_stride})`);
    }
    const loading_point_row = major_direction === 'row' ?  //
        loading_point_div_stride :
        loading_point_mod_stride;
    const loading_point_col = major_direction === 'row' ?  //
        loading_point_mod_stride :
        loading_point_div_stride;
    return {loading_point_row, loading_point_col};
  }

  loadingPointPositionToId(
      loading_point_row: number, loading_point_col: number, major_direction: 'row'|'col', major_stride: number): number;
  loadingPointPositionToId(
      loading_point_row: number|string, loading_point_col: number|string, major_direction: 'row'|'col',
      major_stride: number): number|string;
  loadingPointPositionToId(
      loading_point_row: number|string, loading_point_col: number|string, major_direction: 'row'|'col',
      major_stride: number): number|string {
    const loading_point_row_stride = major_direction === 'row' ? major_stride : 1;
    const loading_point_col_stride = major_direction === 'row' ? 1 : major_stride;

    if ((typeof loading_point_row === 'number') && (typeof loading_point_col === 'number')) {
      return loading_point_row * loading_point_row_stride + loading_point_col * loading_point_col_stride;
    } else {
      const stringValueMulWGSL = (a: string|number, b: number) =>
          ((b === 1) ? `(${a})` :
               (Number.isInteger(Math.log2(b))) ?  //
               `((${a}) << ${Math.log2(b)})` :
               `((${a}) * ${b})`);
      return `${
          stringValueMulWGSL(loading_point_row, loading_point_row_stride) +
          stringValueMulWGSL(loading_point_col, loading_point_col_stride)}`;
    }
  }
}

class VoidCacheHelper extends TSGBlockCacheHelperBase implements TSGBlockCacheHelper {
  constructor(params: TSGBlockCacheHelperConstructorParams) {
    super(params);
  }

  cacheMemoryModuleDefinitionWGSL() {
    return ``;
  }
  cacheMemoryFunctionDefinitionWGSL() {
    return ``;
  }
  cacheMemoryUpdateStatsWGSL() {
    return ``;
  }
  get loadingPointAccessingInCacheExpr() {
    return (row_in_cache: string|number, col_in_cache: string|number, TSG: string) => `${
               this.sourceLoadingPointTSGAccessingExpr(
                   TSG,
                   `(/*row_in_cache*/ i32(${row_in_cache}) + /*loading_point_row_cache_step_TSG_base_WGSL*/ i32(${
                       this.loading_point_row_cache_step_TSG_base_WGSL}))`,
                   `(/*col_in_cache*/ i32(${col_in_cache}) + /*loading_point_col_cache_step_TSG_base_WGSL*/ i32(${
                       this.loading_point_col_cache_step_TSG_base_WGSL}))`)}`;
  }
}

class SharedMemoryCacheHelper extends TSGBlockCacheHelperBase implements TSGBlockCacheHelper {
  constructor(params: TSGBlockCacheHelperConstructorParams) {
    super(params);
  }

  cacheMemoryModuleDefinitionWGSL() {
    return `
const ${this.cache_name}_loading_point_rows_per_cache_step = ${this.loading_points_per_cache_step_TSG.rows}u;
const ${this.cache_name}_loading_point_cols_per_cache_step = ${this.loading_points_per_cache_step_TSG.cols}u;
var<workgroup> ${this.cache_name}: array<array<array<${this.loading_point_WGSL_type}, ${
        this.cache_name}_loading_point_cols_per_cache_step>, ${
        this.cache_name}_loading_point_rows_per_cache_step>, tensor_slice_groups>;
`;
  }

  cacheMemoryFunctionDefinitionWGSL() {
    return ``;
  }

  // workgroupBarrier should be used properly before and after update
  cacheMemoryUpdateStatsWGSL(unfoldStepLoop: boolean = false) {
    const loading_points_number_per_TSG =
        this.loading_points_per_cache_step_TSG.rows * this.loading_points_per_cache_step_TSG.cols;
    const steps = Math.ceil(loading_points_number_per_TSG / this.threads_per_TSG);
    const steps_exactly_divide_loading_points = steps * this.threads_per_TSG === loading_points_number_per_TSG;
    const steps_loop_boundary = unfoldStepLoop ? steps : `${steps}`;
    const loading_points_major_stride = this.loading_points_prefer_major === 'row' ?  //
        this.loading_points_per_cache_step_TSG.cols :
        this.loading_points_per_cache_step_TSG.rows;
    const {loading_point_row, loading_point_col} = this.loadingPointIdToPosition(
        'loading_id_in_cache_step', this.loading_points_prefer_major, loading_points_major_stride);
    return `
// Update shared memory cache for ${this.variable_name}
workgroupBarrier();
{
    let loading_point_row_cache_step_TSG_base: i32 = i32(${this.loading_point_row_cache_step_TSG_base_WGSL});
    let loading_point_col_cache_step_TSG_base: i32 = i32(${this.loading_point_col_cache_step_TSG_base_WGSL});

    ${
        u32LoopUpFrom0WGSL(
            'step', steps_loop_boundary,
            step_var_or_value => `
        // Adjacent threads have adjacent loading_id_in_cache_step
        let loading_id_in_cache_step = thread_id_in_TSG + ${step_var_or_value} * ${this.threads_per_TSG};
        ${
                steps_exactly_divide_loading_points ?  //
                    '/* Steps exactly divide loading points */' :
                    `if (loading_id_in_cache_step < ${loading_points_number_per_TSG})`} {
            let loading_point_row_in_TSG = loading_point_row_cache_step_TSG_base + i32(${loading_point_row});
            let loading_point_col_in_TSG = loading_point_col_cache_step_TSG_base + i32(${loading_point_col});
            ${this.cache_name}[tensor_slice_group][${loading_point_row}][${loading_point_col}] = ${
                this.sourceLoadingPointTSGAccessingExpr(
                    'tensor_slice_group', 'loading_point_row_in_TSG', 'loading_point_col_in_TSG')};
        }`,
            false, 4)}
}
workgroupBarrier();
`;
  }

  get loadingPointAccessingInCacheExpr() {
    return (row_in_cache: string|number, col_in_cache: string|number, TSG: string) =>
               `${this.cache_name}[${TSG}][${row_in_cache}][${col_in_cache}]`;
  }
}

export class SubgroupMemoryCacheHelper extends TSGBlockCacheHelperBase implements TSGBlockCacheHelper {
  subgroup_size: number;
  cache_compute_blocks_per_subgroup: RowsColsSpan;
  loading_point_row_subgroup_cache_step_base_WGSL: string;
  loading_point_col_subgroup_cache_step_base_WGSL: string;
  thread_loading_point_register_mapping_to_subgroup_block: 'adjacent'|'interleaved';

  cache_loading_points_per_subgroup: RowsColsSpan;
  cache_loading_points_number_per_subgroup: number;
  // Loading points for a subgroup is flatten and divided to each thread
  cache_loading_points_per_thread: number;
  cache_loading_points_per_thread_exactly_divide_subgroup: boolean;

  constructor(params: TSGBlockCacheHelperConstructorParams&{
    subgroup_size: number;
    cache_compute_blocks_per_subgroup: RowsColsSpan;
    loading_point_row_subgroup_cache_step_base_WGSL: string;
    loading_point_col_subgroup_cache_step_base_WGSL: string;
    thread_loading_point_register_mapping_to_subgroup_block: 'adjacent'|'interleaved';
  }) {
    super(params);
    this.subgroup_size = params.subgroup_size;
    this.cache_compute_blocks_per_subgroup = params.cache_compute_blocks_per_subgroup;
    this.loading_point_row_subgroup_cache_step_base_WGSL = params.loading_point_row_subgroup_cache_step_base_WGSL;
    this.loading_point_col_subgroup_cache_step_base_WGSL = params.loading_point_col_subgroup_cache_step_base_WGSL;
    this.thread_loading_point_register_mapping_to_subgroup_block =
        params.thread_loading_point_register_mapping_to_subgroup_block;

    this.cache_loading_points_per_subgroup = {
      rows: this.cache_compute_blocks_per_subgroup.rows * this.loading_points_per_compute_block.rows,
      cols: this.cache_compute_blocks_per_subgroup.cols * this.loading_points_per_compute_block.cols,
    };
    this.cache_loading_points_number_per_subgroup =
        this.cache_loading_points_per_subgroup.rows * this.cache_loading_points_per_subgroup.cols;
    this.cache_loading_points_per_thread =
        Math.ceil(this.cache_loading_points_number_per_subgroup / this.subgroup_size);
    this.cache_loading_points_per_thread_exactly_divide_subgroup =
        this.cache_loading_points_per_thread * this.subgroup_size === this.cache_loading_points_number_per_subgroup;
  }

  cacheMemoryModuleDefinitionWGSL() {
    return `
// ${this.cache_name}:
//   cache_loading_points_per_subgroup: rows ${this.cache_loading_points_per_subgroup.rows}, cols ${
        this.cache_loading_points_per_subgroup.cols}
//   cache_loading_points_number_per_subgroup: ${this.cache_loading_points_number_per_subgroup}
const ${this.cache_name}_loading_points_per_thread = ${this.cache_loading_points_per_thread}u;
var<private> ${this.cache_name}: array<${this.loading_point_WGSL_type}, ${this.cache_name}_loading_points_per_thread>;
`;
  }

  cacheMemoryFunctionDefinitionWGSL() {
    return ``;
  }

  subgroupRegisterPositionToSubgroupBlockLoadingPointId(subgroup_id_WGSL: number|string, register_id: number|string):
      number|string {
    switch (this.thread_loading_point_register_mapping_to_subgroup_block) {
      case ('adjacent'): {
        return (
            ((typeof subgroup_id_WGSL === 'number') && (typeof register_id === 'number')) ?
                subgroup_id_WGSL * this.cache_loading_points_per_thread + register_id :
                `(u32(${subgroup_id_WGSL}) * ${this.cache_loading_points_per_thread} + u32(${register_id}))`);
      }
      case ('interleaved'): {
        return (
            ((typeof subgroup_id_WGSL === 'number') && (typeof register_id === 'number')) ?
                subgroup_id_WGSL + register_id * this.subgroup_size :
                `(u32(${subgroup_id_WGSL}) + u32(${register_id}) * ${this.subgroup_size})`);
      }
    }
  }

  // workgroupBarrier should be used properly before and after update
  cacheMemoryUpdateStatsWGSL(unfoldStepLoop: boolean = false) {
    const register_loop_boundary =
        unfoldStepLoop ? this.cache_loading_points_per_thread : `${this.cache_loading_points_per_thread}`;
    const loading_points_major_stride = this.loading_points_prefer_major === 'row' ?  //
        this.cache_loading_points_per_subgroup.cols :
        this.cache_loading_points_per_subgroup.rows;
    const loading_point_row_subgroup_block =
        this.loadingPointIdToPosition(
                'loading_point_id_subgroup', this.loading_points_prefer_major, loading_points_major_stride)
            .loading_point_row;
    const loading_point_col_subgroup_block =
        this.loadingPointIdToPosition(
                'loading_point_id_subgroup', this.loading_points_prefer_major, loading_points_major_stride)
            .loading_point_col;
    return `
// Update subgroup memory cache for ${this.variable_name}
workgroupBarrier();
{
    let loading_point_row_cache_step_TSG_base: i32 = i32(${this.loading_point_row_cache_step_TSG_base_WGSL});
    let loading_point_col_cache_step_TSG_base: i32 = i32(${this.loading_point_col_cache_step_TSG_base_WGSL});

    ${
        u32LoopUpFrom0WGSL(
            'register_id', register_loop_boundary,
            step_var_or_value => `
        let loading_point_id_subgroup = ${
                this.subgroupRegisterPositionToSubgroupBlockLoadingPointId('subgroup_id', step_var_or_value)};
        ${
                this.cache_loading_points_per_thread_exactly_divide_subgroup ?  //
                    '/* Loading points per thread exactly divides subgroup */' :
                    `if (loading_point_id_subgroup < ${this.cache_loading_points_number_per_subgroup})`} {
            let loading_point_row_subgroup_block = ${loading_point_row_subgroup_block};
            let loading_point_col_subgroup_block = ${loading_point_col_subgroup_block};
            let loading_point_row_in_TSG = i32(loading_point_row_subgroup_block) + i32(${
                this.loading_point_row_subgroup_cache_step_base_WGSL}) + i32(loading_point_row_cache_step_TSG_base);
            let loading_point_col_in_TSG = i32(loading_point_col_subgroup_block) + i32(${
                this.loading_point_col_subgroup_cache_step_base_WGSL}) + i32(loading_point_col_cache_step_TSG_base);
            ${this.cache_name}[${step_var_or_value}] =${
                this.sourceLoadingPointTSGAccessingExpr(
                    'tensor_slice_group', 'loading_point_row_in_TSG', 'loading_point_col_in_TSG')};
        }`,
            false, 4)}
}
workgroupBarrier();
`;
  }

  subgroupBlockPositionToThreadAndRegister(
      loading_point_row_in_subgroup_block: number,
      loading_point_col_in_subgroup_block: number): {subgroup_id: number, register_id: number} {
    assert(
        loading_point_row_in_subgroup_block < this.cache_loading_points_per_subgroup.rows,
        `Expect loading_point_row_in_subgroup_block ${loading_point_row_in_subgroup_block} < ${
            this.cache_name}.cache_loading_points_per_subgroup.rows ${this.cache_loading_points_per_subgroup.rows}`);
    assert(
        loading_point_col_in_subgroup_block < this.cache_loading_points_per_subgroup.cols,
        `Expect loading_point_col_in_subgroup_block ${loading_point_col_in_subgroup_block} < ${
            this.cache_name}.cache_loading_points_per_subgroup.cols ${this.cache_loading_points_per_subgroup.cols}`);
    const loading_points_major_stride = this.loading_points_prefer_major === 'row' ?  //
        this.cache_loading_points_per_subgroup.cols :
        this.cache_loading_points_per_subgroup.rows;
    const loading_point_id = this.loadingPointPositionToId(
        loading_point_row_in_subgroup_block, loading_point_col_in_subgroup_block, 'row', loading_points_major_stride);

    switch (this.thread_loading_point_register_mapping_to_subgroup_block) {
      case ('adjacent'): {
        // loading_point_id == subgroup_id * this.cache_loading_points_per_thread + register_id
        return {
          subgroup_id: Math.floor(loading_point_id / this.cache_loading_points_per_thread),
          register_id: loading_point_id % this.cache_loading_points_per_thread,
        };
      }
      case ('interleaved'): {
        // loading_point_id == subgroup_id + register_id * this.subgroup_size
        return {
          subgroup_id: loading_point_id % this.subgroup_size,
          register_id: Math.floor(loading_point_id / this.subgroup_size),
        };
      }
    }
  }

  get loadingPointAccessingInCacheExpr() {
    // Subgroup cache only support constant accessing
    return (row_in_cache: number,
            col_in_cache: number,
            /* TSG omited */) => {
      const {subgroup_id, register_id} = this.subgroupBlockPositionToThreadAndRegister(row_in_cache, col_in_cache);
      return `subgroupBroadcast(${this.cache_name}[${register_id}], ${subgroup_id})`;
    }
  }
}

/*

// Get WGSL expression of all loading points of a input block from given position without bias, the bias can be
// added in the position or in the loading point accessor
const inputBlockLoadingPointsExprWGSLBuilder = (
    loading_points_per_compute_block: RowsColsSpan,
    loadingPointAccessorExpr: (loading_point_row: number|string, loading_point_col: number|string) => string,
    ) =>
    ((
          compute_block_position_row: number|string,
          compute_block_position_col: number|string,
          ): string[][] => {
      const loading_point_row_base = (typeof compute_block_position_row === 'number') ?
          compute_block_position_row * loading_points_per_compute_block.rows :
          `(${compute_block_position_row} * ${loading_points_per_compute_block.rows})`;
      const loading_point_col_base = (typeof compute_block_position_col === 'number') ?
          compute_block_position_col * loading_points_per_compute_block.cols :
          `(${compute_block_position_col} * ${loading_points_per_compute_block.cols})`;
      return (arrayMap(
          loading_points_per_compute_block.rows,
          (row) => arrayMap(
              loading_points_per_compute_block.cols,
              (col) => loadingPointAccessorExpr(
                  `i32(${loading_point_row_base} + ${row})`, `i32(${loading_point_col_base} + ${col})`))));
    });
*/

interface ComputeLoopInputBlockAccessHelper {
  inputALoadingPointsPositionsInCacheFromComputeBlockThread(
      compute_block_thread_M: string|number,
      compute_block_thread_K_inner: string|number): {row_in_cache: string|number, col_in_cache: string|number}[][];
  inputBLoadingPointsPositionsInCacheFromComputeBlockThread(
      compute_block_thread_K_inner: string|number,
      compute_block_thread_N: string|number): {row_in_cache: string|number, col_in_cache: string|number}[][];
}
;
class ComputeLoopInputBlockAccessHelperBase {
  loading_points_per_compute_block: RowsColsSpan;
  constructor(loading_points_per_compute_block: RowsColsSpan) {
    this.loading_points_per_compute_block = loading_points_per_compute_block;
  }
};
// Access helper for TSG shared cache, in which input blocks for a thread must transfrom to its TSG location when
// accessing
class ComputeLoopInputBlockTSGSharedCacheAccessHelper extends ComputeLoopInputBlockAccessHelperBase implements
    ComputeLoopInputBlockAccessHelper {
  constructor(loading_points_per_compute_block: RowsColsSpan) {
    super(loading_points_per_compute_block);
  }
  inputALoadingPointsPositionsInCacheFromComputeBlockThread(
      compute_block_thread_M: string|number,
      compute_block_thread_K_inner: string|number): {row_in_cache: string, col_in_cache: string|number}[][] {
    // Row: compute_block_thread_M --(needed)--> loading_point_TSG_M
    // Col: compute_block_thread_K_inner --(needed)--> loading_point_thread_K_inner --(handled by cache)-->
    // loading_point_TSG_K
    return arrayMap(
        this.loading_points_per_compute_block.rows,
        row => arrayMap(
            this.loading_points_per_compute_block.cols,
            col => ({
              row_in_cache: `((${compute_block_thread_M} + compute_blocks_thread_TSG_base_M) * ${
                  this.loading_points_per_compute_block.rows} + ${row})`,
              col_in_cache: typeof compute_block_thread_K_inner === 'number' ?
                  (compute_block_thread_K_inner * this.loading_points_per_compute_block.cols + col) :
                  `(${compute_block_thread_K_inner} * ${this.loading_points_per_compute_block.cols} + ${col})`,
            })));
  };
  inputBLoadingPointsPositionsInCacheFromComputeBlockThread(
      compute_block_thread_K_inner: string|number,
      compute_block_thread_N: string|number): {row_in_cache: string|number, col_in_cache: string}[][] {
    // Row: compute_block_thread_K_inner --(needed)--> loading_point_thread_K_inner --(handled by void cache)-->
    // loading_point_TSG_K
    // Col: compute_block_thread_N --(needed)--> loading_point_TSG_N
    return arrayMap(
        this.loading_points_per_compute_block.rows,
        row => arrayMap(
            this.loading_points_per_compute_block.cols,
            col => ({
              row_in_cache: typeof compute_block_thread_K_inner === 'number' ?
                  (compute_block_thread_K_inner * this.loading_points_per_compute_block.rows + row) :
                  `(${compute_block_thread_K_inner} * ${this.loading_points_per_compute_block.rows} + ${row})`,
              col_in_cache: `((${compute_block_thread_N} + compute_blocks_thread_TSG_base_N) * ${
                  this.loading_points_per_compute_block.cols} + ${col})`,
            })));
  }
};

// Access helper for subgroup shared cache, in which input blocks for a thread must transfrom to its subgroup block
// location when accessing, and all thread in a subgroup must access the same location at the same time, and the
// accessing position must be number value
class ComputeLoopInputBlockSubgroupSharedCacheAccessHelper extends ComputeLoopInputBlockAccessHelperBase implements
    ComputeLoopInputBlockAccessHelper {
  constructor(loading_points_per_compute_block: RowsColsSpan) {
    super(loading_points_per_compute_block);
  }
  inputALoadingPointsPositionsInCacheFromComputeBlockThread(
      compute_block_thread_M: string|number,
      compute_block_thread_K_inner: string|number): {row_in_cache: number, col_in_cache: number}[][] {
    if (typeof compute_block_thread_M !== 'number') {
      throw new Error(
          `ComputeLoopInputBlockSubgroupSharedCacheAccessHelper requires number compute_block_thread_M, got ${
              compute_block_thread_M}`);
    }
    if (typeof compute_block_thread_K_inner !== 'number') {
      throw new Error(
          `ComputeLoopInputBlockSubgroupSharedCacheAccessHelper requires number compute_block_thread_K_inner, got ${
              compute_block_thread_K_inner}`);
    }
    // Row: compute_block_thread_M --(needed)--> loading_point_TSG_M
    // Col: compute_block_thread_K_inner --(needed)--> loading_point_thread_K_inner --(handled by cache)-->
    // loading_point_TSG_K
    return arrayMap(
        this.loading_points_per_compute_block.rows,
        row => arrayMap(
            this.loading_points_per_compute_block.cols,
            col => ({
              row_in_cache: compute_block_thread_M * this.loading_points_per_compute_block.rows + row,
              col_in_cache: compute_block_thread_K_inner * this.loading_points_per_compute_block.cols + col,
            })));
  };
  inputBLoadingPointsPositionsInCacheFromComputeBlockThread(
      compute_block_thread_K_inner: string|number,
      compute_block_thread_N: string|number): {row_in_cache: number, col_in_cache: number}[][] {
    if (typeof compute_block_thread_K_inner !== 'number') {
      throw new Error(
          `ComputeLoopInputBlockSubgroupSharedCacheAccessHelper requires number compute_block_thread_K_inner, got ${
              compute_block_thread_K_inner}`);
    }
    if (typeof compute_block_thread_N !== 'number') {
      throw new Error(
          `ComputeLoopInputBlockSubgroupSharedCacheAccessHelper requires number compute_block_thread_N, got ${
              compute_block_thread_N}`);
    }
    // Row: compute_block_thread_K_inner --(needed)--> loading_point_thread_K_inner --(handled by void cache)-->
    // loading_point_TSG_K
    // Col: compute_block_thread_N --(needed)--> loading_point_TSG_N
    return arrayMap(
        this.loading_points_per_compute_block.rows,
        row => arrayMap(
            this.loading_points_per_compute_block.cols,
            col => ({
              row_in_cache: compute_block_thread_K_inner * this.loading_points_per_compute_block.rows + row,
              col_in_cache: compute_block_thread_N * this.loading_points_per_compute_block.cols + col,
            })));
  }
};


class InputBlockOptionalRegisterCacheHelper {
  register_cache_enabled: boolean;
  input_name: string;
  loading_points_per_compute_block: RowsColsSpan;
  loading_point_packed_type_WGSL: string;

  register_cache_name: string;
  cached_loading_points_expr_WGSL: string[][];

  constructor(
      register_cache_enabled: boolean, input_name: string, loading_points_per_compute_block: RowsColsSpan,
      loading_point_packed_type_WGSL: string) {
    this.register_cache_enabled = register_cache_enabled;
    this.input_name = input_name;
    this.loading_points_per_compute_block = loading_points_per_compute_block;
    this.loading_point_packed_type_WGSL = loading_point_packed_type_WGSL;

    this.register_cache_name = `register_cache_compute_block_input_${this.input_name}`;
  }
  get defineRegisterCacheWGSL() {
    return this.register_cache_enabled ?
        `// Definition of register cache for input ${this.input_name}
var ${this.register_cache_name}: array<array<${this.loading_point_packed_type_WGSL}, ${
            this.loading_points_per_compute_block.cols}>, ${this.loading_points_per_compute_block.rows}>;` :
        '';
  }
  // Always call updateRegisterCache before using cachedLoadingPointsExprWGSL
  updateRegisterCache(inputBlockLoadingPointsExprWGSL: string[][]) {
    this.cached_loading_points_expr_WGSL = inputBlockLoadingPointsExprWGSL;
    assert(
        this.cached_loading_points_expr_WGSL.length === this.loading_points_per_compute_block.rows,
        `Expect updating register cache for input ${this.input_name} with ${
            this.loading_points_per_compute_block.rows} rows, got ${this.cached_loading_points_expr_WGSL.length}`);
    assert(
        this.cached_loading_points_expr_WGSL[0].length === this.loading_points_per_compute_block.cols,
        `Expect updating register cache for input ${this.input_name} with ${
            this.loading_points_per_compute_block.cols} cols, got ${this.cached_loading_points_expr_WGSL[0].length}`);
    return this.register_cache_enabled ?
        (`// Update register cache for input ${this.input_name}\n` +
         assignLoadingPointsToTargetStats(this.register_cache_name, this.cached_loading_points_expr_WGSL)) :
        '';
  }
  get cachedLoadingPointsExprWGSL() {
    return (
        this.register_cache_enabled ?
            arrayMap(
                this.loading_points_per_compute_block.rows,
                row => arrayMap(
                    this.loading_points_per_compute_block.cols, col => `${this.register_cache_name}[${row}][${col}]`)) :
            this.cached_loading_points_expr_WGSL);
  }
};


export function templatedMatMulProgram(
    op_params: GEMMOperationParameters,
    workgroup_params: GEMMWorkgroupScheduleParameters,
    schedule_params: ShaderScheduleSchemaParameters,
    subgroup_cache_params: ScheduleSchemaSubgroupsParameters = {
      use_subgroups: false
    },
    ): ProgramInfo {
  const {batches_info, input_A, input_B, input_Bias, output, activation, tensor_data_type} = op_params;

  const {output_rows_per_workgroup, output_cols_per_workgroup} = workgroup_params;
  const {compute_schema, tensor_slice} = schedule_params;
  const {tensor_slice_factor} = tensor_slice;

  const has_bias = typeof input_Bias !== 'undefined';

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
  // TODO: Should be parameters or decided by schedule.
  // M-major = adjacent thread has adjacent thread_N_in_TSG
  let buffer_A_cache_type: 'void'|'shared_memory'|'subgroup' =
      (subgroup_cache_params.use_subgroups && subgroup_cache_params.subgroups_shared_on === 'A') ?
      'subgroup' :
      (['void', 'shared_memory', 'subgroup'] as const)[1 as number];
  let buffer_B_cache_type: 'void'|'shared_memory'|'subgroup' =
      (subgroup_cache_params.use_subgroups && subgroup_cache_params.subgroups_shared_on === 'B') ?
      'subgroup' :
      (['void', 'shared_memory', 'subgroup'] as const)[1 as number];

  // const input_A_register_cache_enabled: boolean = buffer_A_cache_type !== 'subgroup';
  // const input_B_register_cache_enabled: boolean = buffer_B_cache_type !== 'subgroup';
  const input_A_register_cache_enabled: boolean = false;
  const input_B_register_cache_enabled: boolean = true;

  // Might be parameter
  const threads_within_TSG_major_M_not_N: boolean = buffer_B_cache_type !== 'subgroup';
  assert(
      (!threads_within_TSG_major_M_not_N) || buffer_B_cache_type !== 'subgroup',
      `Using subgroup cache for input B require !threads_within_TSG_major_M_not_N.`);
  const spatial_loop_order: 'M_outer'|'N_outer' = 'N_outer' as 'M_outer' | 'N_outer';
  // Might be parameter
  const unfold_spatial_M_loop: boolean = buffer_A_cache_type === 'subgroup';
  const unfold_spatial_N_loop: boolean = buffer_B_cache_type === 'subgroup';
  const unfold_K_inner_loop: boolean = subgroup_cache_params.use_subgroups;

  // const compute_block_thread_K_inner_loop_step: number|string = 1;
  const compute_block_thread_K_inner_loop_step: number|string = 8;
  assert(
      workgroup_params.threads_per_workgroup % tensor_slice_factor === 0,
      `Workgroup size ${workgroup_params.threads_per_workgroup} should be a multiple of tensor_slice_factor ${
          tensor_slice_factor}`);
  const threads_per_TSG = workgroup_params.threads_per_workgroup / tensor_slice_factor;
  const threads_along_M_per_TSG = 8;
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
    {name: 'compute_blocks_per_thread_K', type: 'u32'},
  ];
  appendActivationUniforms(activation, uniforms_WGSL_info);

  // Uniforms values
  const programUniforms: ProgramUniform[] = [
    {
      // compute_blocks_per_thread_K
      type: DataType.uint32,
      data: Math.ceil(compute_blocks_per_workgroup.K / tensor_slice_factor)
    },
  ];
  appendActivationUniformsData(activation, programUniforms);
  // Tensor shapes of all input/output variables must be pushed into programUniforms
  assert(
      input_A.buffer_layout.buffer_inner_boundary ===
          ((input_A.buffer_layout.loading_points_layout === 'NHW') ? logical_scalar_size.M : logical_scalar_size.K),
      `Expect for layout ${input_A.buffer_layout.loading_points_layout} \
input_A.buffer_layout.buffer_inner_boundary ${input_A.buffer_layout.buffer_inner_boundary} === \
${
          (input_A.buffer_layout.loading_points_layout === 'NHW') ? `logical_scalar_size.M ${logical_scalar_size.M}` :
                                                                    `logical_scalar_size.K ${logical_scalar_size.K}`}`);
  assert(
      input_A.buffer_layout.buffer_outer_boundary * input_A.buffer_layout.packed_vector_size ===
          (input_A.buffer_layout.loading_points_layout === 'NHW' ? logical_scalar_size.K : logical_scalar_size.M),
      `Expect for layout ${input_A.buffer_layout.loading_points_layout} \
input_A.buffer_layout.buffer_outer_boundary ${input_A.buffer_layout.buffer_outer_boundary} * \
    input_A.buffer_layout.packed_vector_size ${input_A.buffer_layout.packed_vector_size} === \
${
          (input_A.buffer_layout.loading_points_layout === 'NHW') ? `logical_scalar_size.K ${logical_scalar_size.K}` :
                                                                    `logical_scalar_size.M ${logical_scalar_size.M}`}`);
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
${
          (input_B.buffer_layout.loading_points_layout === 'NHW') ? `logical_scalar_size.K ${logical_scalar_size.K}` :
                                                                    `logical_scalar_size.N ${logical_scalar_size.N}`}`);
  assert(
      input_B.buffer_layout.buffer_outer_boundary * input_B.buffer_layout.packed_vector_size ===
          (input_B.buffer_layout.loading_points_layout === 'NHW' ? logical_scalar_size.N : logical_scalar_size.K),
      `Expect for layout ${input_B.buffer_layout.loading_points_layout} \
input_B.buffer_layout.buffer_outer_boundary ${input_B.buffer_layout.buffer_outer_boundary} * \
    input_B.buffer_layout.packed_vector_size ${input_B.buffer_layout.packed_vector_size} === \
${
          (input_B.buffer_layout.loading_points_layout === 'NHW') ? `logical_scalar_size.N ${logical_scalar_size.N}` :
                                                                    `logical_scalar_size.K ${logical_scalar_size.K}`}`);
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
${
          (output.buffer_layout.loading_points_layout === 'NHW') ? `logical_scalar_size.M ${logical_scalar_size.M}` :
                                                                   `logical_scalar_size.N ${logical_scalar_size.N}`}`);
  assert(
      output.buffer_layout.buffer_outer_boundary * output.buffer_layout.packed_vector_size ===
          (output.buffer_layout.loading_points_layout === 'NHW' ? logical_scalar_size.N : logical_scalar_size.M),
      `Expect for layout ${output.buffer_layout.loading_points_layout} \
output.buffer_layout.buffer_outer_boundary ${output.buffer_layout.buffer_outer_boundary} * \
    output.buffer_layout.packed_vector_size ${output.buffer_layout.packed_vector_size} === \
${
          (output.buffer_layout.loading_points_layout === 'NHW') ? `logical_scalar_size.N ${logical_scalar_size.N}` :
                                                                   `logical_scalar_size.M ${logical_scalar_size.M}`}`);
  const output_tensor_loading_points_shape = [
    output.aggregated_batches,
    // output.buffer_layout.loading_points_layout === 'NHW' ? logical_scalar_size.M : logical_scalar_size.N,
    // output.buffer_layout.loading_points_layout === 'NHW' ? logical_scalar_size.N : logical_scalar_size.M,
    output.buffer_layout.buffer_inner_boundary,
    output.buffer_layout.buffer_outer_boundary,
  ];
  const unaggregated_output_tensor_shape = [
    ...batches_info.output_batch_dims, output.buffer_layout.buffer_inner_boundary,
    output.buffer_layout.buffer_outer_boundary * output.buffer_layout.packed_vector_size
  ];
  const input_tensors_shape = [
    input_A_tensor_loading_points_shape, input_B_tensor_loading_points_shape,
    ...(has_bias ? [output_tensor_loading_points_shape] : [])
  ];

  // Push unaggregated batch dims for batch helper
  const batch_helpers_dims = [output_batch_dims, input_A_batch_dims, input_B_batch_dims];
  programUniforms.push(...createTensorShapeVariables(...batch_helpers_dims));
  // Push aggregated inputs and output shape
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
        //   Batch helper variables
        // -----------------------------------------------------------------------------
        const output_batch_helper =
            internalVariable('output_batch_helper', tensor_data_type, output_batch_dims.length, 1);
        const input_A_batch_helper =
            internalVariable('input_A_batch_helper', tensor_data_type, input_A_batch_dims.length, 1);
        const input_B_batch_helper =
            internalVariable('input_B_batch_helper', tensor_data_type, input_B_batch_dims.length, 1);
        // Currently assume bias has the same batches as output
        // const input_Bias_batch_helper = has_bias? internalVariable('input_B_batch_helper',
        // tensor_data_type,input_Bias_batch_dims.length, 1):undefined;

        const batch_helper_variables = [output_batch_helper, input_A_batch_helper, input_B_batch_helper];

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

        // Batch indices helper
        helper_functions.set(
            'ConvertOutputBatchToInputBatch',
            `
// Convert output aggregated batch to vec2(input_A_aggregated_batch, input_B_aggregated_batch)
fn ConvertOutputBatchToInputBatch(output_batch: u32) -> vec2<u32> {
    let batch_indices = ${output_batch_helper.offsetToIndices('output_batch')};

    var input_A_indices: ${input_A_batch_helper.type.indices};
    ${
                u32LoopUpFrom0WGSL(
                    'index_A', input_A_batch_dims.length,
                    (index_A: number) => `
        ${
                        input_A_batch_helper.indicesSet(
                            'input_A_indices', index_A,
                            input_A_broadcasted_dims.includes(index_A) ?
                                0 :
                                output_batch_helper.indicesGet(
                                    'batch_indices', index_A + output_batch_dims.length - input_A_batch_dims.length))}
    `,
                    false, 4)}

    var input_B_indices: ${input_B_batch_helper.type.indices};
    ${
                u32LoopUpFrom0WGSL(
                    'index_B', input_B_batch_dims.length,
                    (index_B: number) => `
        ${
                        input_B_batch_helper.indicesSet(
                            'input_B_indices', index_B,
                            input_B_broadcasted_dims.includes(index_B) ?
                                0 :
                                output_batch_helper.indicesGet(
                                    'batch_indices', index_B + output_batch_dims.length - input_B_batch_dims.length))}
    `,
                    false, 4)}

    let input_A_batch: u32 = ${input_A_batch_helper.indicesToOffset('input_A_indices')};
    let input_B_batch: u32 = ${input_B_batch_helper.indicesToOffset('input_B_indices')};

    return vec2<u32>(input_A_batch, input_B_batch);
}`);

        // Buffer accessing helper functions and expr/stats
        const createLoadBufferHelperFunction = (
            function_name: string,
            buffer_variable: IndicesHelper,
            packed_type_in_buffer_WGSL: string,
            // buffer_layout: BufferLayoutInfo,
            loading_points_layout: 'NHW'|'NWH',
            loading_points_matrix_dims_vec2i_WGSL: string,
            ) => {
          if (buffer_variable.rank !== 3) {
            throw new Error(`Expected buffer A of rank 3, got ${buffer_variable.rank}.`);
          }
          assert(!helper_functions.has(function_name), `Redefining function ${function_name}`);

          // const {loading_points_layout} = buffer_layout;

          helper_functions.set(
              function_name,
              `
// Load a loading point at logical position [batch][row][col] from a matrix buffer.
fn ${function_name}(loading_point_row: i32, loading_point_col: i32, batch: u32) -> ${packed_type_in_buffer_WGSL} {
    var value = ${packed_type_in_buffer_WGSL}(0.0);

    // Handle possible transposed layout
    let inner_dim = ${loading_points_layout === 'NHW' ? 'loading_point_row' : 'loading_point_col'};
    let outer_dim = ${loading_points_layout === 'NHW' ? 'loading_point_col' : 'loading_point_row'};

    if (inner_dim >= 0 && inner_dim < ${loading_points_matrix_dims_vec2i_WGSL}.x && outer_dim >= 0 && outer_dim < ${
                  loading_points_matrix_dims_vec2i_WGSL}.y)
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
            'LoadFromBufferA', input_A_variable, input_A_packed_type_WGSL, input_A.buffer_layout.loading_points_layout,
            'input_A_loading_points_matrix_dims');
        const buffer_B_loading_function_name = createLoadBufferHelperFunction(
            'LoadFromBufferB', input_B_variable, input_B_packed_type_WGSL, input_B.buffer_layout.loading_points_layout,
            'input_B_loading_points_matrix_dims');
        const buffer_Bias_loading_function_name = has_bias ?
            createLoadBufferHelperFunction(
                'LoadFromBufferBias', input_Bias_variable!, output_packed_type_WGSL,
                input_Bias.buffer_layout.loading_points_layout, 'input_Bias_loading_points_matrix_dims') :
            undefined;

        const createStoreBufferHelperFunction = (
            function_name: string,
            buffer_variable: IndicesHelper,
            packed_type_in_buffer_WGSL: string,
            // buffer_layout: BufferLayoutInfo,
            loading_points_layout: 'NHW'|'NWH',
            loading_points_matrix_dims_vec2i_WGSL: string,
            ) => {
          if (buffer_variable.rank !== 3) {
            throw new Error(`Expected buffer A of rank 3, got ${buffer_variable.rank}.`);
          }
          assert(!helper_functions.has(function_name), `Redefining function ${function_name}`);

          // const {loading_points_layout, buffer_inner_boundary, buffer_outer_boundary} = buffer_layout;

          helper_functions.set(
              function_name,
              `
// Store a loading point at logical position [batch][row][col] from a matrix buffer.
fn ${function_name}(value: ${packed_type_in_buffer_WGSL}, loading_point_row: i32, loading_point_col: i32, batch: u32) {
    // Handle possible transposed layout
    let inner_dim = ${loading_points_layout === 'NHW' ? 'loading_point_row' : 'loading_point_col'};
    let outer_dim = ${loading_points_layout === 'NHW' ? 'loading_point_col' : 'loading_point_row'};

    if (inner_dim >= 0 && inner_dim < ${loading_points_matrix_dims_vec2i_WGSL}.x && outer_dim >= 0 && outer_dim < ${
                  loading_points_matrix_dims_vec2i_WGSL}.y)
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
            'StoreToBufferOutput', output_variable, output_packed_type_WGSL, output.buffer_layout.loading_points_layout,
            'output_loading_points_matrix_dims');

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

        const getBufferLoadingPointInTSGExprBuilder =
            (callBufferLoadingFunctionExpr: (loading_point_row: string, loading_point_col: string) => string,
             TSG_global_row_base: string, TSG_global_col_base: string) =>
                ((loading_point_row_in_TSG: number|string, loading_point_col_in_TSG: number|string) =>
                     callBufferLoadingFunctionExpr(
                         `(/*loading_point_row_in_TSG*/ ${loading_point_row_in_TSG} + /*TSG_global_row_base*/ ${
                             TSG_global_row_base})`,
                         `(/*loading_point_col_in_TSG*/ ${loading_point_col_in_TSG} + /*TSG_global_col_base*/ ${
                             TSG_global_col_base})`));

        const getBufferALoadingPointInTSGExpr = getBufferLoadingPointInTSGExprBuilder(
            callBufferALoadingFunctionExpr, 'i32(loading_point_A_TSG_global_row_base)',
            'i32(loading_point_A_TSG_global_col_base)');
        const getBufferBLoadingPointInTSGExpr = getBufferLoadingPointInTSGExprBuilder(
            callBufferBLoadingFunctionExpr, 'i32(loading_point_B_TSG_global_row_base)',
            'i32(loading_point_B_TSG_global_col_base)');
        /*
        const getBufferBiasLoadingPointInTSGExpr = has_bias ?
            getBufferLoadingPointInTSGExprBuilder(
                callBufferBiasLoadingFunctionExpr!, 'i32(loading_point_Output_TSG_global_row_base)',
                'i32(loading_point_Output_TSG_global_col_base)') :
            undefined;
        */

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
let output_loading_point_thread_global_base_row = (compute_blocks_thread_TSG_base_M + compute_blocks_workgroup_global_base_M) * ${
              loading_points_per_compute_block_Output.rows};
let output_loading_point_thread_global_base_col = (compute_blocks_thread_TSG_base_N + compute_blocks_workgroup_global_base_N) * ${
              loading_points_per_compute_block_Output.cols};
var output_loading_point_block_global_base_row = output_loading_point_thread_global_base_row;
${
              u32LoopUpFrom0WGSL(
                  'compute_block_thread_M', compute_blocks_per_thread_M,
                  (compute_block_thread_M: string|number) => `
    // Begin compute_block_thread_M loop
    var output_loading_point_block_global_base_col = output_loading_point_thread_global_base_col;
    ${
                      u32LoopUpFrom0WGSL(
                          'compute_block_thread_N', compute_blocks_per_thread_N,
                          (compute_block_thread_N: string|number) => `
        // Begin compute_block_thread_N loop
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
            ${buffer_Output_storing_function_name}(*output_vector, i32(output_loading_points_global_row), i32(output_loading_points_global_col), batch_output);
        }`).join('\n')).join('\n')}
        output_loading_point_block_global_base_col += ${loading_points_per_compute_block_Output.cols};
        // End compute_block_thread_N loop`,
                          false,
                          /* outer_ident_width */ 4)}
    output_loading_point_block_global_base_row += ${loading_points_per_compute_block_Output.rows};
    // End compute_block_thread_M loop`)}
`;
        };

        // -----------------------------------------------------------------------------
        //   Cache hierarchy
        // -----------------------------------------------------------------------------
        let buffer_A_cache: TSGBlockCacheHelper;
        let compute_loop_input_A_block_access_helper: ComputeLoopInputBlockAccessHelper;
        let buffer_B_cache: TSGBlockCacheHelper;
        let compute_loop_input_B_block_access_helper: ComputeLoopInputBlockAccessHelper;

        switch (buffer_A_cache_type) {
          case ('void'): {
            buffer_A_cache = new VoidCacheHelper({
              variable_name: 'input_A',
              threads_per_TSG,
              loading_points_per_compute_block: loading_points_per_compute_block_A,
              cache_compute_blocks_per_TSG:
                  {rows: compute_blocks_per_workgroup.M, cols: compute_block_thread_K_inner_loop_step},
              loading_point_WGSL_type: input_A_packed_type_WGSL,
              loading_point_row_cache_step_TSG_base_WGSL: '0',
              loading_point_col_cache_step_TSG_base_WGSL: 'cache_step_A_loading_point_col_in_TSG_base',
              sourceLoadingPointTSGAccessingExpr: (_: string, row_in_TSG: string, col_in_TSG: string) =>
                  getBufferALoadingPointInTSGExpr(row_in_TSG, col_in_TSG),
              loading_points_prefer_major: 'row',
            });
            compute_loop_input_A_block_access_helper =
                new ComputeLoopInputBlockTSGSharedCacheAccessHelper(loading_points_per_compute_block_A);
            break;
          }
          case ('shared_memory'): {
            buffer_A_cache = new SharedMemoryCacheHelper({
              variable_name: 'input_A',
              threads_per_TSG,
              loading_points_per_compute_block: loading_points_per_compute_block_A,
              cache_compute_blocks_per_TSG:
                  {rows: compute_blocks_per_workgroup.M, cols: compute_block_thread_K_inner_loop_step},
              loading_point_WGSL_type: input_A_packed_type_WGSL,
              loading_point_row_cache_step_TSG_base_WGSL: '0',
              loading_point_col_cache_step_TSG_base_WGSL: 'cache_step_A_loading_point_col_in_TSG_base',
              sourceLoadingPointTSGAccessingExpr: (_: string, row_in_TSG: string, col_in_TSG: string) =>
                  getBufferALoadingPointInTSGExpr(row_in_TSG, col_in_TSG),
              loading_points_prefer_major: 'row',
            });
            compute_loop_input_A_block_access_helper =
                new ComputeLoopInputBlockTSGSharedCacheAccessHelper(loading_points_per_compute_block_A);
            break;
          }
          case ('subgroup'): {
            if (!subgroup_cache_params.use_subgroups) {
              throw new Error(`Using subgroup cache without valid subgroup_cache_params.`);
            }
            // Ensure all threads in a subgroup have the same thread_M_in_TSG and thus access the same input row block
            assert(
                threads_within_TSG_major_M_not_N,
                'Using subgroup cache for input A requires threads_within_TSG_major_M_not_N.');
            assert(
                threads_along_N_per_TSG % subgroup_cache_params.expected_subgroup_size === 0,
                `Using subgroup cache for input A requires threads_along_N_per_TSG ${
                    threads_along_N_per_TSG} can be divided by expected_subgroup_size ${
                    subgroup_cache_params.expected_subgroup_size}.`);

            const cache_compute_blocks_per_subgroup: RowsColsSpan = {
              rows: compute_blocks_per_thread_M,
              cols: compute_block_thread_K_inner_loop_step,
            };

            buffer_A_cache = new SubgroupMemoryCacheHelper({
              variable_name: 'input_A',
              threads_per_TSG,
              loading_points_per_compute_block: loading_points_per_compute_block_A,
              cache_compute_blocks_per_TSG:
                  {rows: compute_blocks_per_workgroup.M, cols: compute_block_thread_K_inner_loop_step},
              loading_point_WGSL_type: input_A_packed_type_WGSL,
              loading_point_row_cache_step_TSG_base_WGSL: '0',
              loading_point_col_cache_step_TSG_base_WGSL: 'cache_step_A_loading_point_col_in_TSG_base',
              sourceLoadingPointTSGAccessingExpr: (_: string, row_in_TSG: string, col_in_TSG: string) =>
                  getBufferALoadingPointInTSGExpr(row_in_TSG, col_in_TSG),
              loading_points_prefer_major: 'row',
              subgroup_size: subgroup_cache_params.expected_subgroup_size,
              cache_compute_blocks_per_subgroup,
              loading_point_row_subgroup_cache_step_base_WGSL:
                  `thread_M_in_TSG * ${compute_blocks_per_thread_M} * ${loading_points_per_compute_block_A.rows}`,
              loading_point_col_subgroup_cache_step_base_WGSL: '0',
              thread_loading_point_register_mapping_to_subgroup_block:
                  subgroup_cache_params.thread_loading_point_register_mapping_to_subgroup_block,
            });
            compute_loop_input_A_block_access_helper =
                new ComputeLoopInputBlockSubgroupSharedCacheAccessHelper(loading_points_per_compute_block_A);
            break;
          }
        }

        switch (buffer_B_cache_type) {
          case ('void'): {
            buffer_B_cache = new VoidCacheHelper({
              variable_name: 'input_B',
              threads_per_TSG,
              loading_points_per_compute_block: loading_points_per_compute_block_B,
              cache_compute_blocks_per_TSG:
                  {rows: compute_block_thread_K_inner_loop_step, cols: compute_blocks_per_workgroup.N},
              loading_point_WGSL_type: input_B_packed_type_WGSL,
              loading_point_row_cache_step_TSG_base_WGSL: 'cache_step_B_loading_point_row_in_TSG_base',
              loading_point_col_cache_step_TSG_base_WGSL: '0',
              sourceLoadingPointTSGAccessingExpr: (/*TSG*/ _: string, row_in_TSG: string, col_in_TSG: string) =>
                  getBufferBLoadingPointInTSGExpr(row_in_TSG, col_in_TSG),
              loading_points_prefer_major: 'row',
            });
            compute_loop_input_B_block_access_helper =
                new ComputeLoopInputBlockTSGSharedCacheAccessHelper(loading_points_per_compute_block_B);
            break;
          }
          case ('shared_memory'): {
            buffer_B_cache = new SharedMemoryCacheHelper({
              variable_name: 'input_B',
              threads_per_TSG,
              loading_points_per_compute_block: loading_points_per_compute_block_B,
              cache_compute_blocks_per_TSG:
                  {rows: compute_block_thread_K_inner_loop_step, cols: compute_blocks_per_workgroup.N},
              loading_point_WGSL_type: input_B_packed_type_WGSL,
              loading_point_row_cache_step_TSG_base_WGSL: 'cache_step_B_loading_point_row_in_TSG_base',
              loading_point_col_cache_step_TSG_base_WGSL: '0',
              sourceLoadingPointTSGAccessingExpr: (/*TSG*/ _: string, row_in_TSG: string, col_in_TSG: string) =>
                  getBufferBLoadingPointInTSGExpr(row_in_TSG, col_in_TSG),
              loading_points_prefer_major: 'row',
            });
            compute_loop_input_B_block_access_helper =
                new ComputeLoopInputBlockTSGSharedCacheAccessHelper(loading_points_per_compute_block_B);
            break;
          }

          case ('subgroup'): {
            if (!subgroup_cache_params.use_subgroups) {
              throw new Error(`Using subgroup cache without valid subgroup_cache_params.`);
            }
            // Ensure all threads in a subgroup have the same thread_N_in_TSG and thus access the same input col block
            assert(
                !threads_within_TSG_major_M_not_N,
                'Using subgroup cache for input B requires !threads_within_TSG_major_M_not_N.');
            assert(
                threads_along_M_per_TSG % subgroup_cache_params.expected_subgroup_size === 0,
                `Using subgroup cache for input B requires threads_along_M_per_TSG ${
                    threads_along_M_per_TSG} can be divided by expected_subgroup_size ${
                    subgroup_cache_params.expected_subgroup_size}.`);

            const cache_compute_blocks_per_subgroup: RowsColsSpan = {
              rows: compute_block_thread_K_inner_loop_step,
              cols: compute_blocks_per_thread_N,
            };

            buffer_B_cache = new SubgroupMemoryCacheHelper({
              variable_name: 'input_B',
              threads_per_TSG,
              loading_points_per_compute_block: loading_points_per_compute_block_B,
              cache_compute_blocks_per_TSG:
                  {rows: compute_block_thread_K_inner_loop_step, cols: compute_blocks_per_workgroup.N},
              loading_point_WGSL_type: input_B_packed_type_WGSL,
              loading_point_row_cache_step_TSG_base_WGSL: 'cache_step_B_loading_point_row_in_TSG_base',
              loading_point_col_cache_step_TSG_base_WGSL: '0',
              sourceLoadingPointTSGAccessingExpr: (_: string, row_in_TSG: string, col_in_TSG: string) =>
                  getBufferALoadingPointInTSGExpr(row_in_TSG, col_in_TSG),
              loading_points_prefer_major: 'row',
              subgroup_size: subgroup_cache_params.expected_subgroup_size,
              cache_compute_blocks_per_subgroup,
              loading_point_row_subgroup_cache_step_base_WGSL: '0',
              loading_point_col_subgroup_cache_step_base_WGSL:
                  `thread_N_in_TSG * ${compute_blocks_per_thread_N} * ${loading_points_per_compute_block_B.cols}`,
              thread_loading_point_register_mapping_to_subgroup_block:
                  subgroup_cache_params.thread_loading_point_register_mapping_to_subgroup_block,
            });
            compute_loop_input_B_block_access_helper =
                new ComputeLoopInputBlockSubgroupSharedCacheAccessHelper(loading_points_per_compute_block_A);
            break;
          }
        }

        const inputABlockLoadingPointsFromCacheWithComputeLoopVarOrValueExprWGSL =
            (compute_block_thread_M_var_or_value: string|number,
             compute_block_thread_K_inner_var_or_value: string|number) =>
                compute_loop_input_A_block_access_helper
                    .inputALoadingPointsPositionsInCacheFromComputeBlockThread(
                        compute_block_thread_M_var_or_value, compute_block_thread_K_inner_var_or_value)
                    .map(
                        positionsInRow => positionsInRow.map(
                            position => buffer_A_cache.loadingPointAccessingInCacheExpr(
                                position.row_in_cache, position.col_in_cache, 'tensor_slice_group')));
        const inputBBlockLoadingPointsFromCacheWithComputeLoopVarOrValueExprWGSL =
            (compute_block_thread_K_inner_var_or_value: string|number,
             compute_block_thread_N_var_or_value: string|number) =>
                compute_loop_input_B_block_access_helper
                    .inputBLoadingPointsPositionsInCacheFromComputeBlockThread(
                        compute_block_thread_K_inner_var_or_value, compute_block_thread_N_var_or_value)
                    .map(
                        positionsInRow => positionsInRow.map(
                            position => buffer_B_cache.loadingPointAccessingInCacheExpr(
                                position.row_in_cache, position.col_in_cache, 'tensor_slice_group')));

        const InputABlockOptionalRegisterCacheHelper = new InputBlockOptionalRegisterCacheHelper(
            input_A_register_cache_enabled, 'A', loading_points_per_compute_block_A, input_A_packed_type_WGSL);
        const InputBBlockOptionalRegisterCacheHelper = new InputBlockOptionalRegisterCacheHelper(
            input_B_register_cache_enabled, 'B', loading_points_per_compute_block_B, input_B_packed_type_WGSL);


        // -----------------------------------------------------------------------------
        //   Nesting compute loops
        // -----------------------------------------------------------------------------
        const nesting_compute_loop_valid_loop_var_names =
            ['compute_block_thread_M', 'compute_block_thread_N', 'compute_block_thread_K_inner'] as const;
        type NestingComputeLoopValidLoopVarName = typeof nesting_compute_loop_valid_loop_var_names[number];
        type NestingComputeLoopValidLoopVarsOrValues = {[K in NestingComputeLoopValidLoopVarName]?: string | number};
        interface NestingComputeLoop {
          loop_var_name_WGSL: NestingComputeLoopValidLoopVarName;
          loop_upper_boundary: string|number;
          loop_body_before_inner_loop: (valid_loop_vars_or_values: NestingComputeLoopValidLoopVarsOrValues) => string;
          loop_body_after_inner_loop: (valid_loop_vars_or_values: NestingComputeLoopValidLoopVarsOrValues) => string;
          disable_unfold?: boolean;
          additional_ident_for_inner_loop?: number;
        }
        interface NestingComputeLoopDeepestBody {
          deepest_body: (valid_loop_vars_or_values: NestingComputeLoopValidLoopVarsOrValues) => string;
        }
        const buildNestingComputeLoop =
            (loops: (NestingComputeLoop|NestingComputeLoopDeepestBody)[],
             valid_loop_vars_or_values: NestingComputeLoopValidLoopVarsOrValues): string => {
              const current_loop = loops[0];
              const remain_loops = loops.slice(1);
              if ('deepest_body' in current_loop) {
                return current_loop.deepest_body(valid_loop_vars_or_values);
              }
              const wrapLoopBodyFunction: () => ((loop_var_or_value: string|number) => string) = () => {
                return (loop_var_or_value: string|number) => {
                  const updated_loop_vars_or_values: NestingComputeLoopValidLoopVarsOrValues =
                      {...valid_loop_vars_or_values, [current_loop.loop_var_name_WGSL]: loop_var_or_value};
                  const loop_body_before_inner_loop =
                      current_loop.loop_body_before_inner_loop(updated_loop_vars_or_values);
                  const loop_body_after_inner_loop =
                      current_loop.loop_body_after_inner_loop(updated_loop_vars_or_values);
                  return addIdent(
                      [
                        loop_body_before_inner_loop,
                        addIdent(
                            buildNestingComputeLoop(remain_loops, updated_loop_vars_or_values),
                            current_loop.additional_ident_for_inner_loop ?? 0, 'addAllLines'),
                        loop_body_after_inner_loop
                      ].join(''),
                      4, 'addAllLines');
                }
              };
              return u32LoopUpFrom0WGSL(
                  current_loop.loop_var_name_WGSL, current_loop.loop_upper_boundary, wrapLoopBodyFunction(),
                  current_loop.disable_unfold);
            };

        const compute_block_thread_K_inner_loop: NestingComputeLoop = {
          loop_var_name_WGSL: 'compute_block_thread_K_inner',
          loop_upper_boundary: compute_block_thread_K_inner_loop_step,
          loop_body_before_inner_loop: (valid_loop_vars_or_values: NestingComputeLoopValidLoopVarsOrValues) => `
// Begin of compute_block_thread_K_inner loop${
              typeof valid_loop_vars_or_values.compute_block_thread_K_inner === 'number' ?
                  `, step ${valid_loop_vars_or_values.compute_block_thread_K_inner} of ${
                      compute_block_thread_K_inner_loop_step}` :
                  ''}
// Compute block K is unbiased from thread to TSG
let compute_block_thread_K_in_TSG = compute_block_thread_K_outer_loop + ${
              valid_loop_vars_or_values.compute_block_thread_K_inner!};
// Compute block K is unbiased from workgroup to global
let compute_block_K_global_biased = compute_block_thread_K_in_TSG + compute_blocks_TSG_workgroup_base_K;
if (compute_block_thread_K_in_TSG < uniforms.compute_blocks_per_thread_K) {
`,
          loop_body_after_inner_loop: () => `
    // End of condition (compute_block_thread_K_in_TSG < uniforms.compute_blocks_per_thread_K)
}
// End of compute_block_thread_K_inner loop`,
          disable_unfold: !unfold_K_inner_loop,
          additional_ident_for_inner_loop: 4,
        };

        const compute_block_thread_M_loop: NestingComputeLoop = {
          loop_var_name_WGSL: 'compute_block_thread_M',
          loop_upper_boundary: compute_blocks_per_thread_M,
          loop_body_before_inner_loop: (valid_loop_vars_or_values: NestingComputeLoopValidLoopVarsOrValues) => `
// Begin of compute_block_thread_M loop${
              typeof valid_loop_vars_or_values.compute_block_thread_M === 'number' ?
                  `, step ${valid_loop_vars_or_values.compute_block_thread_M} of ${compute_blocks_per_thread_M}` :
                  ''}
let compute_block_M_TSG_biased = ${
              valid_loop_vars_or_values.compute_block_thread_M!} + compute_blocks_thread_TSG_base_M;
${InputABlockOptionalRegisterCacheHelper.defineRegisterCacheWGSL}
${
              InputABlockOptionalRegisterCacheHelper.updateRegisterCache(
                  inputABlockLoadingPointsFromCacheWithComputeLoopVarOrValueExprWGSL(
                      valid_loop_vars_or_values.compute_block_thread_M!,
                      valid_loop_vars_or_values.compute_block_thread_K_inner!))}
`,
          loop_body_after_inner_loop: () => `
// End of compute_block_thread_M loop`,
          disable_unfold: !unfold_spatial_M_loop,
        };

        const compute_block_thread_N_loop: NestingComputeLoop = {
          loop_var_name_WGSL: 'compute_block_thread_N',
          loop_upper_boundary: compute_blocks_per_thread_N,
          loop_body_before_inner_loop: (valid_loop_vars_or_values: NestingComputeLoopValidLoopVarsOrValues) => `
// Begin of compute_block_thread_N loop${
              typeof valid_loop_vars_or_values.compute_block_thread_N === 'number' ?
                  `, step ${valid_loop_vars_or_values.compute_block_thread_N} of ${compute_blocks_per_thread_N}` :
                  ''}
let compute_block_N_TSG_biased = ${
              valid_loop_vars_or_values.compute_block_thread_N!} + compute_blocks_thread_TSG_base_N;
${InputBBlockOptionalRegisterCacheHelper.defineRegisterCacheWGSL}
${
              InputBBlockOptionalRegisterCacheHelper.updateRegisterCache(
                  inputBBlockLoadingPointsFromCacheWithComputeLoopVarOrValueExprWGSL(
                      valid_loop_vars_or_values.compute_block_thread_K_inner!,
                      valid_loop_vars_or_values.compute_block_thread_N!))}
`,
          loop_body_after_inner_loop: () => `
// End of compute_block_thread_N loop`,
          disable_unfold: !unfold_spatial_N_loop,
        };

        const compute_loop_deepest_body: NestingComputeLoopDeepestBody = {
          deepest_body: (valid_loop_vars_or_values: NestingComputeLoopValidLoopVarsOrValues) => `
// Compute block
${
              calcComputeBlockStats(
                  InputABlockOptionalRegisterCacheHelper.cachedLoadingPointsExprWGSL,
                  InputBBlockOptionalRegisterCacheHelper.cachedLoadingPointsExprWGSL,
                  `acc_array[${valid_loop_vars_or_values.compute_block_thread_M}][${
                      valid_loop_vars_or_values.compute_block_thread_N}]`,
                  0, 0, compute_schema)}
`
        };

        const nesting_compute_loops: (NestingComputeLoop|NestingComputeLoopDeepestBody)[] = [
          compute_block_thread_K_inner_loop,
          ...(spatial_loop_order === 'M_outer' ? [compute_block_thread_M_loop, compute_block_thread_N_loop] :
                                                 [compute_block_thread_N_loop, compute_block_thread_M_loop]),
          compute_loop_deepest_body
        ];


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
// const compute_blocks_per_workgroup_K = ${compute_blocks_per_workgroup.K}u;
const compute_blocks_per_workgroup_N = ${compute_blocks_per_workgroup.N}u;

const A_loading_point_rows_per_compute_block = ${loading_points_per_compute_block_A.rows}u;
const A_loading_point_cols_per_compute_block = ${loading_points_per_compute_block_A.cols}u;
const B_loading_point_rows_per_compute_block = ${loading_points_per_compute_block_B.rows}u;
const B_loading_point_cols_per_compute_block = ${loading_points_per_compute_block_B.cols}u;
const Output_loading_point_rows_per_compute_block = ${loading_points_per_compute_block_Output.rows}u;
const Output_loading_point_cols_per_compute_block = ${loading_points_per_compute_block_Output.cols}u;

const threads_along_M_per_TSG = ${threads_along_M_per_TSG}u;
const threads_along_N_per_TSG = ${threads_along_N_per_TSG}u;
const_assert(threads_along_M_per_TSG * threads_along_N_per_TSG == threads_per_TSG);

// compute_blocks_per_thread_K = ceil(compute_blocks_per_workgroup_K / tensor_slice_factor) comes from uniform
// to allow more shader reusing.
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

${
            shader_helper.registerUniforms(uniforms_WGSL_info)
                .registerInternalVariables(...batch_helper_variables)
                .declareVariables(...input_variables, output_variable)}

// Workgroup-cached uniforms
var<workgroup> input_A_loading_points_matrix_dims: vec2<i32>;
var<workgroup> input_B_loading_points_matrix_dims: vec2<i32>;
${has_bias ? 'var<workgroup> input_Bias_loading_points_matrix_dims: vec2<i32>;' : ''}
var<workgroup> output_loading_points_matrix_dims: vec2<i32>;

fn WorkgroupInit() {
  input_A_loading_points_matrix_dims = vec2<i32>(${input_A_variable.shape}.yz);
  input_B_loading_points_matrix_dims = vec2<i32>(${input_B_variable.shape}.yz);
  ${has_bias ? `input_Bias_loading_points_matrix_dims = vec2<i32>(${input_Bias_variable!.shape}.yz);` : ''}
  output_loading_points_matrix_dims = vec2<i32>(${output_variable.shape}.yz);
}

// Invocation scope variables
var<private> tensor_slice_group: u32;
var<private> thread_M_in_TSG: u32;
var<private> thread_N_in_TSG: u32;
var<private> batch_output: u32;
var<private> batch_A: u32;
var<private> batch_B: u32;
var<private> compute_blocks_thread_TSG_base_M: u32;
// Dimension K is unbiased from thread to TSG
// const compute_blocks_thread_TSG_base_K = 0;
var<private> compute_blocks_thread_TSG_base_N: u32;
var<private> loading_point_A_TSG_global_row_base: u32;
var<private> loading_point_A_TSG_global_col_base: u32;
var<private> loading_point_B_TSG_global_row_base: u32;
var<private> loading_point_B_TSG_global_col_base: u32;
var<private> loading_point_Output_TSG_global_row_base: u32;
var<private> loading_point_Output_TSG_global_col_base: u32;

var<private> acc_array: array<array<ComputeBlockOutputType, compute_blocks_per_thread_N>, compute_blocks_per_thread_M>;

// Helper functions
${[...helper_functions.values()].join('\n\n')}

${
            tensor_slice_factor > 1 ? `
// Shared memory for adding up tensor sliced results
var<workgroup> tensor_slice_acc: array<array<array<ComputeBlockOutputType, compute_blocks_per_workgroup_N>, compute_blocks_per_workgroup_M>, ${
                                          Math.ceil(tensor_slice_factor / 2)}>;

// Helper function for adding up tensor slice results
fn TensorSliceResultsMergingStep(remaining_slices: u32) {
    let half_slices = (remaining_slices+1) / 2;  // ceil(remaining_slices / 2)
    // Sub-step A: Store upper-half reg to lower-half SM
    if ((tensor_slice_group < remaining_slices) && (tensor_slice_group >= half_slices)) {
        for (var compute_block_thread_M: u32 = 0; compute_block_thread_M < compute_blocks_per_thread_M; compute_block_thread_M++) {
            for (var compute_block_thread_N: u32 = 0; compute_block_thread_N < compute_blocks_per_thread_N; compute_block_thread_N++) {
                let reg_acc = &acc_array[compute_block_thread_M][compute_block_thread_N];
                tensor_slice_acc
                    [tensor_slice_group-half_slices]
                    [compute_blocks_thread_TSG_base_M+compute_block_thread_M]
                    [compute_blocks_thread_TSG_base_N+compute_block_thread_N] =
                    *reg_acc;
            }
        }
    }
    workgroupBarrier();
    // Sub-step B: Lower-half add SM into reg
    if (tensor_slice_group < half_slices) {
        for (var compute_block_thread_M: u32 = 0; compute_block_thread_M < compute_blocks_per_thread_M; compute_block_thread_M++) {
            for (var compute_block_thread_N: u32 = 0; compute_block_thread_N < compute_blocks_per_thread_N; compute_block_thread_N++) {
                let workgroup_acc =
                      &tensor_slice_acc
                          [tensor_slice_group]
                          [compute_blocks_thread_TSG_base_M+compute_block_thread_M]
                          [compute_blocks_thread_TSG_base_N+compute_block_thread_N];
                let reg_acc = &acc_array[compute_block_thread_M][compute_block_thread_N];
                ${addIdent(addOutputBlocksStats('(*reg_acc)', '(*workgroup_acc)'), 16, 'keepFirstLine')}
            }
        }
    }
    workgroupBarrier();
}
` :
                                      ''}

// Buffer cache global definition, if any
${buffer_A_cache.cacheMemoryModuleDefinitionWGSL()}
${buffer_B_cache.cacheMemoryModuleDefinitionWGSL()}

@compute @workgroup_size(WGS, 1, 1)
fn main(
    @builtin(workgroup_id) wid: vec3u,
    @builtin(local_invocation_id) lid: vec3u,
    ${subgroup_cache_params.use_subgroups ? '@builtin(subgroup_invocation_id) subgroup_id: u32,' : ''}
    ${subgroup_cache_params.use_subgroups ? '@builtin(subgroup_size) subgroup_size: u32,' : ''}
) {
    if(all(lid == vec3u())) {
      WorkgroupInit();
    }
    workgroupBarrier();

    let compute_blocks_workgroup_global_base_M = wid.x * compute_blocks_per_workgroup_M;
    // compute_blocks_workgroup_global_base_K == 0
    let compute_blocks_workgroup_global_base_N = wid.y * compute_blocks_per_workgroup_N;
    batch_output = wid.z;
    let input_batches = ConvertOutputBatchToInputBatch(batch_output);
    batch_A = input_batches.x;
    batch_B = input_batches.y;

    let thread_id_in_workgroup = lid.x;
    tensor_slice_group = ${tensor_slice_factor === 1 ? '0u' : 'thread_id_in_workgroup / threads_per_TSG'};
    let thread_id_in_TSG = ${
            tensor_slice_factor === 1 ? 'thread_id_in_workgroup' : 'thread_id_in_workgroup % threads_per_TSG'};
    // Threads arranged in M major manner within a TSG, could also be N major.
    // const threads_within_TSG_major_M_not_N = ${threads_within_TSG_major_M_not_N};
    // thread_M_in_TSG = select(thread_id_in_TSG / threads_along_N_per_TSG, thread_id_in_TSG % threads_along_M_per_TSG, threads_within_TSG_major_M_not_N);
    // thread_N_in_TSG = select(thread_id_in_TSG % threads_along_N_per_TSG, thread_id_in_TSG / threads_along_M_per_TSG, threads_within_TSG_major_M_not_N);
    thread_M_in_TSG = ${
            threads_within_TSG_major_M_not_N ? 'thread_id_in_TSG / threads_along_N_per_TSG' :
                                               'thread_id_in_TSG % threads_along_M_per_TSG'};
    thread_N_in_TSG = ${
            threads_within_TSG_major_M_not_N ? 'thread_id_in_TSG % threads_along_N_per_TSG' :
                                               'thread_id_in_TSG / threads_along_M_per_TSG'};

    loading_point_A_TSG_global_row_base = compute_blocks_workgroup_global_base_M * A_loading_point_rows_per_compute_block;
    loading_point_A_TSG_global_col_base = tensor_slice_group * uniforms.compute_blocks_per_thread_K * A_loading_point_cols_per_compute_block;
    loading_point_B_TSG_global_row_base = tensor_slice_group * uniforms.compute_blocks_per_thread_K * B_loading_point_rows_per_compute_block;
    loading_point_B_TSG_global_col_base = compute_blocks_workgroup_global_base_N * B_loading_point_cols_per_compute_block;
    loading_point_Output_TSG_global_row_base = compute_blocks_workgroup_global_base_M * Output_loading_point_rows_per_compute_block;
    loading_point_Output_TSG_global_col_base = compute_blocks_workgroup_global_base_N * Output_loading_point_cols_per_compute_block;

    compute_blocks_thread_TSG_base_M = thread_M_in_TSG * compute_blocks_per_thread_M;
    compute_blocks_thread_TSG_base_N = thread_N_in_TSG * compute_blocks_per_thread_N;

    let compute_blocks_TSG_workgroup_base_K = tensor_slice_group * uniforms.compute_blocks_per_thread_K;

    // Buffer cache function-scope definition, if any
    ${buffer_A_cache.cacheMemoryFunctionDefinitionWGSL()}
    ${buffer_B_cache.cacheMemoryFunctionDefinitionWGSL()}

    const compute_block_thread_K_inner_loop_step: u32 = ${compute_block_thread_K_inner_loop_step};

    for (
        var compute_block_thread_K_outer_loop: u32 = 0;
        compute_block_thread_K_outer_loop < uniforms.compute_blocks_per_thread_K;
        compute_block_thread_K_outer_loop += compute_block_thread_K_inner_loop_step
    ) {
        // Handle cache steps along K, if any
        let cache_step_A_loading_point_col_in_TSG_base = i32(compute_block_thread_K_outer_loop) * ${
            loading_points_per_compute_block_A.cols};
        let cache_step_B_loading_point_row_in_TSG_base = i32(compute_block_thread_K_outer_loop) * ${
            loading_points_per_compute_block_B.rows};
        ${addIdent(buffer_A_cache.cacheMemoryUpdateStatsWGSL(), 8, 'keepFirstLine')}
        ${addIdent(buffer_B_cache.cacheMemoryUpdateStatsWGSL(), 8, 'keepFirstLine')}

        ${addIdent(buildNestingComputeLoop(nesting_compute_loops, {}), 8)}
    }

    ${
            tensor_slice_factor > 1 ? `
    // Add up tensor slice results
    ${(() => {
              let code = '';
              for (let remaining_slices = tensor_slice_factor; remaining_slices > 1;
                   remaining_slices = Math.ceil(remaining_slices / 2)) {
                code += `
    // Merge tensor slices ${remaining_slices} -> ${Math.ceil(remaining_slices / 2)}
    TensorSliceResultsMergingStep(${remaining_slices});`;
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
    output_compute_blocks_per_workgroup: {M: compute_blocks_per_workgroup.M, N: compute_blocks_per_workgroup.N},
    tensor_slice_factor,
    input_A_layout: input_A.buffer_layout.loading_points_layout,
    input_B_layout: input_B.buffer_layout.loading_points_layout,
    output_layout: output.buffer_layout.loading_points_layout,
    packing_size: {
      input_A: input_A.buffer_layout.packed_vector_size,
      input_B: input_B.buffer_layout.packed_vector_size,
      output: output.buffer_layout.packed_vector_size,
    },
    batch_indices: {
      input_A_broadcasted_dims,
      input_B_broadcasted_dims,
    },
  };

  // LOG(`fatal`, `templatedMatMulProgram Return`);

  return {
    name: `MatMulTemplatedTrival-${
        Object.entries(cacheKey).map((entry) => entry.map(x => JSON.stringify(x)).join(':')).join('-')};`,
    shaderCache: {
      // hint: `${activation.activation};${Object.entries(workgroup_params).map((entry) =>
      // entry.join(':')).join(';')};`,
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
            packed_vector_direction: loading_points_layout === 'NHW' ? 'row' : 'col',
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
        output_rows_per_workgroup: 32,
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
          tensor_slice_factor: 1,
          // tensor_slice_factor: 4,
          // tensor_slice_factor: 2,
          // tensor_slice_input_policy: 'continious',
        },
      };

      const subgroup_cache_params: ScheduleSchemaSubgroupsParameters = {
        use_subgroups: true,
        subgroups_shared_on: 'A',
        expected_subgroup_size: 16,
        // thread_loading_point_register_mapping_to_subgroup_block: 'adjacent',
        thread_loading_point_register_mapping_to_subgroup_block: 'interleaved',
      };

      LOG(`fatal`, `Before templatedMatMulProgram`);

      return templatedMatMulProgram(op_params, workgroup_params, schedule_params, subgroup_cache_params);
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
