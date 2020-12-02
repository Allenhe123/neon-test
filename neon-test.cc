#include <iostream>
#include <chrono>
#include <arm_neon.h>

using namespace std;

////////////////////////////////////////////////////////
uint64_t Now() {
    auto now  = std::chrono::high_resolution_clock::now();
    auto nano_time_pt = std::chrono::time_point_cast<std::chrono::nanoseconds>(now);
    auto epoch = nano_time_pt.time_since_epoch();
    uint64_t now_nano = std::chrono::duration_cast<std::chrono::nanoseconds>(epoch).count();
    return now_nano;
}

////////////////////////////////////////////////////////
float sum_array(float *arr, int len)
{
    if(NULL == arr || len < 1)
    {
        cout<<"input error\n";
        return 0;
    }
    float sum(0.0);
    for(int i=0; i<len; ++i)
    {
        sum += *arr++;
    }
    return sum;
}

////////////////////////////////////////////////////////
float sum_array_neon(float *arr, int len)
{
    if(NULL == arr || len < 1)
    {
        cout<<"input error\n";
        return 0;
    }
 
    int dim4 = len >> 2; // 数组长度除4整数
    int left4 = len & 3; // 数组长度除4余数
    float32x4_t sum_vec = vdupq_n_f32(0.0);//定义用于暂存累加结果的寄存器且初始化为0
    for (; dim4>0; dim4--, arr+=4) //每次同时访问4个数组元素
    {
        float32x4_t data_vec = vld1q_f32(arr); //依次取4个元素存入寄存器vec
        sum_vec = vaddq_f32(sum_vec, data_vec);//ri = ai + bi 计算两组寄存器对应元素之和并存放到相应结果
    }
    //将累加结果寄存器中的所有元素相加得到最终累加值
    float sum = vgetq_lane_f32(sum_vec, 0)+vgetq_lane_f32(sum_vec, 1)+vgetq_lane_f32(sum_vec, 2)+vgetq_lane_f32(sum_vec, 3);
    for (; left4>0; left4--, arr++)
        sum += (*arr) ;   //对于剩下的少于4的数字，依次计算累加即可
    return sum;
}


/////////////////////////////////////////////////////////////

int main() {
	size_t size = 1280 * 736;
	float temp[1280 * 736];
	for (int i=0; i<size; i++) {
		temp[i] = i;
	}

	auto t0 = Now();
	sum_array(temp, size);
	auto t1 = Now();

	cout << "T1: " << t1 - t0 << endl;

	auto t2 = Now();
	sum_array_neon(temp, size);
	auto t3 = Now();

	cout << "T2: " << t3 - t2 << endl;

	return 0;
}
