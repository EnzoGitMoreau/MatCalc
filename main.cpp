
#include <stdio.h>
#include <stdlib.h>
#include "blas.h"
#include "matsym.h"
#include <iostream>
#include <cblas.h>
#include <random>
#include "matrix.h"
#include "blockMatrix.hpp"
#include <time.h>
#include "blockInstance.hpp"
#include "omp.h"
#include <deque> 
#include <tuple>
#include "real.h"
#include "sparmatsymblk.h"

#define NUMBER 1*2*3*4*5*6*7*2

typedef std::tuple<int,int,int,Matrix44*> Tuple2;
typedef std::deque<Tuple2> Queue2;
void matrix_multiply_4x4_neonMain(const real* __restrict__ X1, const real* __restrict__ X2,real* __restrict__ Y1,  real* __restrict__ Y2, int blocksize, int matsize, const real* __restrict__ valptr) {
        // these are the columns A
    
        //Matrice 4x4
        
        float64x2_t A01;
        float64x2_t A02;
        float64x2_t A11;
        float64x2_t A12;
        float64x2_t A21;
        float64x2_t A22;
        float64x2_t A31;
        float64x2_t A32;
        float64x2_t X01;
        float64x2_t X02;
        float64x2_t X11;
        float64x2_t X12;
        float64x2_t Y21;

        float64x2_t Y22;
        
float64x2_t partialSum;
float64x2_t partialSum2;
float64x2_t accumulator1;
float64x2_t accumulator2;
        X01 = vld1q_f64(X1);
        X02 = vld1q_f64(X1+2);
        X11 = vld1q_f64(X2);
        X12 = vld1q_f64(X2+2);


        
        

//accumulator1 = {0,0};
//accumulator2 = {0,0};
        accumulator1 =vld1q_f64(Y1);
        accumulator2 = vld1q_f64(Y1+2);

        
        

        A01 = vld1q_f64(valptr);
        A02 = vld1q_f64(valptr+2);
        A11 = vld1q_f64(valptr+matsize);
A21 = vld1q_f64(valptr+2*matsize);
A31 = vld1q_f64(valptr+3*matsize);
        accumulator1 =vfmaq_n_f64(accumulator1, A01, vgetq_lane_f64(X01,0));
        Y21 = vld1q_f64(Y2);
        Y22 = vld1q_f64(Y2+2);
        partialSum = vmulq_f64(A01, X11);
        
        
        
        
        
        
        accumulator2 =vfmaq_n_f64(accumulator2, A02, vgetq_lane_f64(X01,0));
        partialSum = vfmaq_f64(partialSum, A02, X12);



        
        accumulator1 =vfmaq_n_f64(accumulator1, A11, vgetq_lane_f64(X01,1));
        partialSum2 =vmulq_f64(A11, X11);



        A12 = vld1q_f64(valptr+matsize+2);
        accumulator2 =vfmaq_n_f64(accumulator2, A12, vgetq_lane_f64(X01,1));
        partialSum2 = vfmaq_f64(partialSum2, A12, X12);
        ///Finishing calcul
        Y21 =vzip1q_f64(partialSum, partialSum2);
        partialSum = vzip2q_f64(partialSum, partialSum2);
        Y21 = vaddq_f64(partialSum, Y21);
        Y21 = vaddq_f64(vld1q_f64( Y2), Y21);
        vst1q_f64(Y2, Y21);

        
        accumulator1 =vfmaq_n_f64(accumulator1, A21, vgetq_lane_f64(X02,0));
        partialSum = vmulq_f64( A21, X11);

        A22 = vld1q_f64(valptr+2*matsize+2);
        accumulator2 =vfmaq_n_f64(accumulator2, A22, vgetq_lane_f64(X02,0));
        partialSum = vfmaq_f64(partialSum, A22, X12);
        


        
        accumulator1 =vfmaq_n_f64(accumulator1, A31, vgetq_lane_f64(X02,1));
        partialSum2 =vmulq_f64(A31, X11);

        A32 = vld1q_f64(valptr+3*matsize+2);
        accumulator2 =vfmaq_n_f64(accumulator2, A32, vgetq_lane_f64(X02,1));
        partialSum2 = vfmaq_f64(partialSum2, A32, X12);
        //Finishing calcul
        Y22 =vzip1q_f64(partialSum, partialSum2);
        partialSum = vzip2q_f64(partialSum, partialSum2);
        Y22 = vaddq_f64(partialSum, Y22);
        Y22 = vaddq_f64(vld1q_f64( Y2+2), Y22);
vst1q_f64(Y2+2, Y22);
vst1q_f64(Y1,accumulator1);
vst1q_f64(Y1+2,accumulator2);
        
        
    
}
//Idées : arrondir la matrice au multiple de la taille du block le plus proche



void printMatrixValues(MatrixSymmetric* matrix, bool only_int = true)
{
    char str[32];
    real* val = matrix->data();
    std::cout << "\nMatrix of size : "<< matrix->size()<<"\n";

    for(int i=0; i<matrix->size(); i++)
    {
        
        
        for(int j = 0; j<matrix->size();j++)
        {
        if(!only_int){
        snprintf(str, sizeof(str), "%9.2f", *matrix->addr(i,j));
        }
        else
        {
            std::snprintf(str,sizeof(str), "%d ", (int)*matrix->addr(i,j));
        }
        std::cout << str <<"" ;
        }
        if(i!=matrix->size()-1)
        {
            std::cout<<"\n";
        }
       
    }
}
void setMatrixRandomValues(MatrixSymmetric matrix)
{
    #ifdef VERBOSE
    std::cout << "Setting random values";
    #endif
    try{

    real* val = matrix.data();
        
#ifdef VERBOSE
    std::cout << "Matrix of size : "<< matrix.size()<<"\n";
#endif
    for(int i=0; i<matrix.size(); i++)
    {
    
        for(int j = 0; j<=i;j++)
        {
                ///real value = dis(gen);
                if(i==j)
                {
                    val[i*matrix.size()+j] = 0;
                }
                else
                {
                    if(true)
                    {
                        val[i*matrix.size()+j] =(i+j)+1/(i+j);
                        val[j*matrix.size()+i] =(i+j)+1/(i+j);
                    }
                    else
                    { val[i*matrix.size()+j] = 0;}
                    
                }
                //val[i*matrix.size() + j] = (i+j)%100;
                //val[j*matrix.size() + i] = (i+j)%100;
         
        }
    }
    }
    catch(std::exception){
        std::cout << "Error loading RNG";
    }
    
}


void fillSMSB(int nbBlocks, int matsize,int blocksize, SparMatSymBlk* matrix)
{
    
    
    
    
    for(int i =0; i<matsize/blocksize;i++)
    {
      
        for(int j=0; j<=i; j++)
        {
            Matrix44* blockTest = new Matrix44(1,1,3,4,5,6,7,8,9,1,2,12,13,14,15,16);
            matrix->block(i,j).add_full(*blockTest);
        }
    }
}


void calculate(std::deque<Queue2*>** workingPhases2, int phase_number, real*X, real*Y, int big_mat_size)
{
    
    real* Y1 = (real*) malloc(big_mat_size * sizeof(real));
    int blocksize = 4;
    int matsize= blocksize;
    real* Y2 = (real*) malloc(big_mat_size * sizeof(real));
    for(int k = 0; k<phase_number; k++)//On calcule chaque phase
    {
        std::deque<Queue2*>* phase_wblocks = workingPhases2[k];
        while(!phase_wblocks->empty())
        {
            Queue2* block_of_work = phase_wblocks->front();
            phase_wblocks->pop_front();
            while(!block_of_work->empty())
            {
                Tuple2 bloc_to_calculate = block_of_work->front();
                block_of_work->pop_front();
                std::cout<<"Calculating block : ("<<std::get<0>(bloc_to_calculate)<<","<<std::get<1>(bloc_to_calculate)<<","<<std::get<2>(bloc_to_calculate)<<") at adress: "<<std::get<3>(bloc_to_calculate)<<"\n";
                
                int index_x =std::get<0>(bloc_to_calculate);
                int index_y = std::get<1>(bloc_to_calculate);
                int swap = std::get<2>(bloc_to_calculate);
                real* valptr1 = std::get<3>(bloc_to_calculate)->val;
#ifdef VERBOSE_2
                std::cout<<"Consuming : "<<"("<<index_x<<","<<index_y<<","<<swap<<")"<<"\n";
#endif
                //int t = std::max((index_x)*blocksize,index_y * blocksize) +matsize *std::min((index_x)*blocksize,index_y * blocksize);
                //const real* valptr1 =(valptr +t);
                const real* X1 = X+ (index_y * blocksize);
                const real* X2 = X + index_x * blocksize;
                real* Y1_ = Y1 + index_x * blocksize;
                real* Y2_  = Y2 + index_y * blocksize;
                if(swap == 1)
                {
                    const real* temp;
                    Y2_ = Y1 +index_y * blocksize; //Y2 "becomes" Y1
                    Y1_ = Y2 +index_x * blocksize;
                    
                  
                }
                if(index_x != index_y)
                {
                    float64x2_t A01;
                    float64x2_t A02;
                    float64x2_t A11;
                    float64x2_t A12;
                    float64x2_t A21;
                    float64x2_t A22;
                    float64x2_t A31;
                    float64x2_t A32;
                    
                    float64x2_t X01;
                    float64x2_t X02;
                    float64x2_t X11;
                    float64x2_t X12;
                    float64x2_t Y21;
                    
                    float64x2_t Y22;
                    
                    float64x2_t partialSum;
                    float64x2_t partialSum2;
                    float64x2_t accumulator1;
                    float64x2_t accumulator2;
                    X01 = vld1q_f64(X1);
                    X02 = vld1q_f64(X1+2);
                    X11 = vld1q_f64(X2);
                    X12 = vld1q_f64(X2+2);
                    
                    
                    
                    
                    
                    //accumulator1 = {0,0};
                    //accumulator2 = {0,0};
                    accumulator1 =vld1q_f64(Y1_);
                    accumulator2 = vld1q_f64(Y1_+2);
                    
                    
                    
                    
                    A01 = vld1q_f64(valptr1);
                    A02 = vld1q_f64(valptr1+2);
                    A11 = vld1q_f64(valptr1+matsize);
                    A21 = vld1q_f64(valptr1+2*matsize);
                    A31 = vld1q_f64(valptr1+3*matsize);
                    accumulator1 =vfmaq_n_f64(accumulator1, A01, vgetq_lane_f64(X01,0));
                    Y21 = vld1q_f64(Y2);
                    Y22 = vld1q_f64(Y2+2);
                    partialSum = vmulq_f64(A01, X11);
                    
                    
                    
                    
                    
                    
                    accumulator2 =vfmaq_n_f64(accumulator2, A02, vgetq_lane_f64(X01,0));
                    partialSum = vfmaq_f64(partialSum, A02, X12);
                    
                    
                    
                    
                    accumulator1 =vfmaq_n_f64(accumulator1, A11, vgetq_lane_f64(X01,1));
                    partialSum2 =vmulq_f64(A11, X11);
                    
                    
                    
                    A12 = vld1q_f64(valptr1+matsize+2);
                    accumulator2 =vfmaq_n_f64(accumulator2, A12, vgetq_lane_f64(X01,1));
                    partialSum2 = vfmaq_f64(partialSum2, A12, X12);
                    ///Finishing calcul
                    Y21 =vzip1q_f64(partialSum, partialSum2);
                    partialSum = vzip2q_f64(partialSum, partialSum2);
                    Y21 = vaddq_f64(partialSum, Y21);
                    Y21 = vaddq_f64(vld1q_f64( Y2_), Y21);
                    vst1q_f64(Y2_, Y21);
                    
                    
                    accumulator1 =vfmaq_n_f64(accumulator1, A21, vgetq_lane_f64(X02,0));
                    partialSum = vmulq_f64( A21, X11);
                    
                    A22 = vld1q_f64(valptr1+2*matsize+2);
                    accumulator2 =vfmaq_n_f64(accumulator2, A22, vgetq_lane_f64(X02,0));
                    partialSum = vfmaq_f64(partialSum, A22, X12);
                    
                    
                    
                    
                    accumulator1 =vfmaq_n_f64(accumulator1, A31, vgetq_lane_f64(X02,1));
                    partialSum2 =vmulq_f64(A31, X11);
                    
                    A32 = vld1q_f64(valptr1+3*matsize+2);
                    accumulator2 =vfmaq_n_f64(accumulator2, A32, vgetq_lane_f64(X02,1));
                    partialSum2 = vfmaq_f64(partialSum2, A32, X12);
                    //Finishing calcul
                    Y22 =vzip1q_f64(partialSum, partialSum2);
                    partialSum = vzip2q_f64(partialSum, partialSum2);
                    Y22 = vaddq_f64(partialSum, Y22);
                    Y22 = vaddq_f64(vld1q_f64( Y2_+2), Y22);
                    vst1q_f64(Y2_+2, Y22);
                    vst1q_f64(Y1_,accumulator1);
                    vst1q_f64(Y1_+2,accumulator2);
                    
                    
                }
                else
                {
                    //matrix->matrix_multiply_4x4_neonMid(Vec+(index_x* blocksize),Y1+index_y * blocksize, blocksize, matsize, valptr1);
                    X1 = X+(index_x* blocksize);
                    Y1_ = Y1+index_y * blocksize;
                    
                    float64x2_t A01;
                    float64x2_t A02;
                    float64x2_t A11;
                    float64x2_t A12;
                    float64x2_t A21;
                    float64x2_t A22;
                    float64x2_t A31;
                    float64x2_t A32;
                    float64x2_t X01;
                    float64x2_t X02;
                    
                    float64x2_t Y11;
                    float64x2_t Y12;
                    
                    X01 = vld1q_f64(X1);
                    
                    Y11 = vld1q_f64(Y1_);
                    Y12  = vld1q_f64(Y1_+2);
                    
                    A01 = vld1q_f64(valptr1);
                    Y11=vfmaq_laneq_f64(Y11, A01, X01,0);
                    
                    A02 = vld1q_f64(valptr1+2);
                    Y12=vfmaq_laneq_f64(Y12, A02, X01,0);
                    
                    A11 = vld1q_f64(valptr1+matsize);
                    Y11=vfmaq_laneq_f64(Y11, A11, X01,1);
                    
                    A12 = vld1q_f64(valptr1+matsize+2);
                    Y12=vfmaq_laneq_f64(Y12, A12, X01,1);
                    
                    X02 = vld1q_f64(X1+2);
                    A21 = vld1q_f64(valptr1+2*matsize);
                    Y11=vfmaq_laneq_f64(Y11, A21, X02,0);
                    A22 = vld1q_f64(valptr1+2*matsize+2);
                    Y12=vfmaq_laneq_f64(Y12, A22, X02,0);
                    
                    A31 = vld1q_f64(valptr1+3*matsize);
                    Y11=vfmaq_laneq_f64(Y11, A31, X02,1);
                    A32 = vld1q_f64(valptr1+3*matsize+2);
                    Y12=vfmaq_laneq_f64(Y12, A32, X02,1);
                    
                    vst1q_f64(Y1_,Y11);
                    vst1q_f64(Y1_+2,Y12);
                }
            }
        }
        
        
        
    }
    for(int i=0;i<big_mat_size;i++)
    {
        Y[i]+= Y1[i]+Y2[i];
    }
    
}

        
    

int main(int argc, char* argv[])
{

    
    int blocksize = 4;
    size_t size = blocksize * NUMBER;
    size = 4*8*2*2*2*2*2*2*2;
    real* Vec = (real*)malloc(size * sizeof(real));//Defining vector to do MX
    real* Y_res = (real*)malloc(size * sizeof(real));//Y
    real* Y_true = (real*)malloc(size * sizeof(real));//Y_true to compare
    real* Y_third = (real*)malloc(size * sizeof(real));//Y_true to compare
    real* Y_dif = (real*)malloc(size * sizeof(real));//Y_diff that will store differences
    real* Y_diff = (real*)malloc(size * sizeof(real));//Y_diff that will store differences
    for(int i=0; i<size;i++)
    {
        Vec[i]=i;//Init
        Y_res[i] = 0;
    }



  
    SparMatSymBlk testMatrix = SparMatSymBlk();
    testMatrix.allocate(size);
    testMatrix.resize(size);
    testMatrix.reset();
    
    //testMatrix.element(1,2) += bl ockTest.value(1,2);
    fillSMSB(5, size, 4, &testMatrix);
    //testMatrix.printSummary(std::cout, 0, 25);
    testMatrix.prepareForMultiply(1);
    



    int nMatrix = 10000;
    int nThreads = 1;
    testMatrix.prepareForMultiply(1);
    
    //testMatrix.vecMulAddMt(8, Vec, Y_res);
    //testMatrix.testFullCalcMtNtime3(nThreads, Vec, Y_res,nMatrix);
    testMatrix.testFullCalcMtNtime2(nThreads, Vec, Y_res,nMatrix);
    
    //testMatrix.testFullCalcMt(8, Vec, Y_res);

    for(int i=0; i<nMatrix;i++)
    {
        testMatrix.vecMulAdd(Vec, Y_true);
    }
    int nbDiff = 0;
    
    for(int i=0; i<size;i++)
    {
        Y_dif[i] = Y_true[i] - Y_res[i];
        if(Y_dif[i]!=0)
        {
            nbDiff++;
        }
        //Y_dift[i] = Y_true[i] - Y_third[i];
        
    }

if(nbDiff !=0)
{
    std::cout<<"Resultat computation originelle\n";
    for(int i =0; i< size; i++)
    {
        std::cout<<Y_true[i]<<" ";
    }
    std::cout<<"Resultat computation maison\n";
    for(int i =0; i< size; i++)
    {
        std::cout<<Y_res[i]<<" ";
    }
    std::cout<<"\n\nDifference of true_computation\n";
    for(int i=0; i<size;i++)
    {
        std::cout<<Y_dif[i]<<" ";
    }
}
else
{
        std::cout<<"\nComputation went well";
}

#if 0

    //blockTest.print(std::cout);

    int nbrMatrix = 1;
    for(int i=0; i<nbrMatrix; i++)
    {
   
        MatrixSymmetric test(size);
        
        setMatrixRandomValues(test);
        //printMatrixValues(&test);
        
      
        
        for(int i=0; i<size;i++)
        {
            Y_true[i] = 0;//init
        }
        //clock_t start_1 = clock();
        //Y_third =  multi(&test,Vec,blocksize,true);
        
        //clock_t end_1 = clock();
        
        //clock_t start_2 = clock();
        test.vecMulAdd(Vec, Y_true);//calculate using boths techniques
        //clock_t end_2 = clock();
        
        //clock_t start_3 = clock();
        //test.vecMulAdd(Vec, Y_true);//calculate using boths techniques
        //cblas_dsymv(CblasRowMajor, CblasLower, 4, 1.0, test.data(), size, Vec, 1, 1.0, Y_true, 1);
        
        //cblas_dsymv(CblasRowMajor, CblasLower, 4, 1.0, test.data(), size, Vec, 1, 1.0, Y_true, 1);
        int t1 = std::time(NULL);
        
        test.vecMulPerBlock(Vec, Y_res, 4, 8);
        std::cout<<"Matrix calculation performed time : "<< t1 - std::time(NULL);
        //Y_res= multi(&test,Vec,blocksize,false);//calculate using both techniques
        //clock_t end_3 = clock();
        //const real* val = test.data();
       // matrix_multiply_4x4_neonMain(Vec, Vec, Y_thidrd, Y_diff, 4, size, val);
       //matrix_multiply_4x4_neonMain(Vec, Vec, Y_third, Y_diff, 4, size, val);
      
    }
    


 


    std::cout<<"\n\nResult of block_computation\n";
    for(int i=0; i<size;i++)
    {
        std::cout<<Y_res[i]<<" ";
    }

  

    cout<<"\n\nResult of third_computation\n";
    for(int i=0; i<size;i++)
    {
        cout<<Y_third[i]<<" ";
    }
    cout<<"\n\nDifference of true_computation with third\n";
    for(int i=0; i<size;i++)
    {
        cout<<Y_dift[i]<<" ";
    }

    auto t_1 = 1000*(end_1 - start_1)/CLOCKS_PER_SEC;
    auto t_2 = 1000*(end_2 - start_2)/CLOCKS_PER_SEC;
    auto t_3 = 1000*(end_3 - start_3)/CLOCKS_PER_SEC;
    std::cout<<"\n\nElapses times\nBlock Algorithm:"<<t_1<<"\nNative algorithm:"<<t_2<<"\n"<<"fullvecmut"<<t_3;
    


    using namespace std::chrono;
    auto start = high_resolution_clock::now();
    std::cout<<"Perf test for bothvecmul\n";
    
    real R1[4] = {1,2,3,4};
    real R2[4] = {1,2,3,4};
    real Y1[4] = {0,0,0,0};
    real Y2[4] = {0,0,0,0};
    int c1 =0;

    real Y12[4] = {0,0,0,0};
    real Y22[4] = {0,0,0,0};
    
    int c2 = 0;
    Matrix44* blockTest = new Matrix44(1,1,3,4,5,6,7,8,9,1,2,12,13,14,15,16);
  
        
    blockTest -> bothvecmulopti(R1, R2, Y12, Y22);
    
    blockTest-> bothvecmul(R1, R2, Y1, Y2);
    
    std::cout<<"Results:";
    for(int i=0; i<4;i++)
    {
        std::cout<<Y1[i]<<" ";
    }
    std::cout<<" \n";
    for(int i=0; i<4;i++)
    {
        std::cout<<Y2[i]<<" ";
    }
    std::cout<<" \n\n";
    for(int i=0; i<4;i++)
    {
        std::cout<<Y12[i]<<" ";
    }
    std::cout<<" \n";
    for(int i=0; i<4;i++)
    {
        std::cout<<Y22[i]<<" ";
    }

    std::cout<<"\nNumber executed for 1:"<<c1<<"\n";
    std::cout<<"Number executed for 2:"<<c2<<"\n";

    
#endif
    //ding of parallel region
}
