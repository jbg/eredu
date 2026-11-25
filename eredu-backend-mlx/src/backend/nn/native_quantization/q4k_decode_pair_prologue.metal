uint lane = thread_position_in_grid.x;
uint projection = thread_position_in_grid.y % 2u;
uint output_pair = thread_position_in_grid.y / 2u;
const device uchar* weight = projection == 0u ? gate_weight : up_weight;
uint local_group = thread_position_in_threadgroup.y;
threadgroup float products[SIMD_GROUPS * ROWS_PER_SIMD];
uint route = 0u;
uint group = 0u;
