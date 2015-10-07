

__kernel void solve_odd(__global float *odd, __global float *even, int height, int width,float beta,float omega){
  int tx = get_global_id(0);
  int ty = get_global_id(1);
  int index = tx*height+ty;

  if((tx + ty)%2 != 0){
      if(tx > 0 && ty > 0 && tx < width-1 && ty < height-1){
          odd[index] = (1.0-omega)*odd[index] + omega/(2*(1+beta))
                     *(even[index+1] + even[index-1] + beta*(even[index+height] + even[index-height]));
      }
  }
}

__kernel void solve_even(__global float *odd, __global float *even, int height, int width,float beta,float omega){
	int tx = get_global_id(0);
	int ty = get_global_id(1);

	int index = tx*height+ty;

	if((tx + ty)%2 == 0){
        if(tx > 0 && ty > 0 && tx < width-1 && ty < height-1){
            even[index] = (1.0-omega)*even[index] + omega/(2*(1+beta))
                        *(odd[index+1] + odd[index-1] + beta*(odd[index+height] + odd[index-height]));
        }
    }
}

__kernel void merge_oddeven(__global float *odd, __global float *even, int height, int width){

  int tx = get_global_id(0);
  int ty = get_global_id(1);
  int index = tx*height+ty;

  if((tx + ty)%2 == 0 && tx < width-1 && ty < height-1){
      odd[index] = even[index];
  }

}