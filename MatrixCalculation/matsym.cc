// Cytosim was created by Francois Nedelec. Copyright 2020 Cambridge University.

#include "matsym.h"
#include "assert_macro.h"
#include <cblas.h>
#include <arm_neon.h>
#include "blockInstance.hpp"

MatrixSymmetric::MatrixSymmetric()
{
    allocated_ = 0;
    val        = nullptr;
    in_charge  = true;
}


void MatrixSymmetric::allocate(size_t alc)
{
    if ( alc > allocated_ )
    {
        allocated_ = alc;
        free_real(val);
        val = new_real(alc*alc);
    }
}


void MatrixSymmetric::deallocate()
{
    if ( in_charge )
        free_real(val);
    allocated_ = 0;
    val = nullptr;
}


void MatrixSymmetric::reset()
{
    for ( size_t i = 0; i < size_ * size_; ++i )
        val[i] = 0;
}


void MatrixSymmetric::scale( real alpha )
{
    for ( size_t i = 0; i < size_ * size_; ++i )
        val[i] *= alpha;
}

//------------------------------------------------------------------------------
real& MatrixSymmetric::operator()( size_t x, size_t y)
{
    assert_true( x < size_ );
    assert_true( y < size_ );
    return val[ std::max(x,y) + dimension_ * std::min(x,y) ];
}


real* MatrixSymmetric::addr( size_t x, size_t y) const
{
    assert_true( x < size_ );
    assert_true( y < size_ );
    return val + ( std::max(x,y) + dimension_ * std::min(x,y) );
}


bool MatrixSymmetric::notZero() const
{
    return true;
}


size_t MatrixSymmetric::nbElements(size_t start, size_t stop) const
{
    assert_true( start <= stop );
    stop = std::min(stop, size_);
    start = std::min(start, size_);

    return size_ * ( stop - start );
}


std::string MatrixSymmetric::what() const
{
    return "full-symmetric";
}


void MatrixSymmetric::vecMulAddBlock2(const real * __restrict__ X, real* __restrict__ Y, int index_x, int index_y, int blocksize, int matsize) const
{
    
    real* ptr1 =(val + std::max((index_x)*blocksize,index_y * blocksize) +matsize * std::min((index_x)*blocksize,index_y * blocksize));
    Y += blocksize*index_x;
    for (int i = 0; i < blocksize; i++)
    {
                    real tempX = X[i];
                    real tempY = 0;
                    for (int j = i + 1; j < blocksize; ++j) {
                        real val  =ptr1[j+ i*matsize];
                        Y[j] += tempX * val;
                        tempY +=  val * X[j];
                    }
        Y[i]+= tempY+tempX * ptr1[i +i*matsize];;
    }
   
            
            
        
    
}
void MatrixSymmetric::transVecMulAddBlock(const real* X, real*Y, int index_x, int index_y, int blocksize, int matsize) const
{
    for ( size_t i = 0; i < blocksize; ++i )
    {
        real value = 0;
        for ( size_t j = 0; j <blocksize; ++j )
        {
            
            value += *(val + std::min((index_x)*blocksize + i,index_y * blocksize + j) +dimension_ * std::max((index_x)*blocksize + i,index_y * blocksize + j)) * X[j];
        }
        Y[i+blocksize*index_x] += value;
    }
}

void MatrixSymmetric:: transVecMulAddBlock3(const real* __restrict__ X1, const real* __restrict__ X2,real* __restrict__ Y1,  real* __restrict__ Y2, int blocksize, int matsize, const real* __restrict__ valptr ) const
{
    for (int i=0;i<blocksize;i++)
    {
        real tempX = X2[i];//Y2 seems fine
        real tempY = 0;
        for (int j = 0; j < blocksize; ++j) {
            real val  = valptr[j+i*matsize];
            Y2[j] += tempX * val;
            tempY +=  val * X1[j];
           //add prefetch intrinsics
        }
    Y1[i]+= tempY;
    }
    
}


void MatrixSymmetric:: transVecMulAddBlock4(const real* __restrict__ X1, const real* __restrict__ X2,real* __restrict__ Y1,  real* __restrict__ Y2, int blocksize, int matsize, const real* __restrict__ valptr ) const
{
    //We will assume, block are 4x4 blocks
    for (int i=0;i<blocksize;i++)
    {
        real tempX = X2[i];//Y2 seems fine
        real tempY = 0;
        for (int j = 0; j < blocksize; ++j) {
            real val  = valptr[j+i*matsize];
            Y2[j] += tempX * val;
            tempY +=  val * X1[j];
           //add prefetch intrinsics
        }
    Y1[i]+= tempY;
    }
    
}
void MatrixSymmetric::matrix_multiply_4x4_neon(const real* __restrict__ X1, const real* __restrict__ X2,real* __restrict__ Y1,  real* __restrict__ Y2, int blocksize, int matsize, const real* __restrict__ valptr) const{
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
            float64x2_t Y11;
            float64x2_t Y12;
            float64x2_t Y22;
            float64x2_t TEMP_1;
            
    float64x2_t accumulator1;
    float64x2_t accumulator2;
            X01 = vld1q_f64(X1);
            X02 = vld1q_f64(X1+2);
            X11 = vld1q_f64(X2);
            X12 = vld1q_f64(X2+2);
   

            
            Y11 = vld1q_f64(Y1);
            Y12  = vld1q_f64(Y1+2);
   
    //accumulator1 = {0,0};
    //accumulator2 = {0,0};
    accumulator1 =vld1q_f64(Y2);
    accumulator2 = vld1q_f64(Y2+2);
    
            Y21 = vld1q_f64(Y2);
            Y22 = vld1q_f64(Y2+2);
            TEMP_1 = {0,0};
            TEMP_1 =vzip1q_f64(Y21, TEMP_1);
    
            A01 = vld1q_f64(valptr);
            vcopyq_laneq_f64(TEMP_1,0,Y21,0);
            TEMP_1 = vfmaq_f64(TEMP_1, A01, X11);//On garde les
           
            Y11=vfmaq_laneq_f64(Y11, A01, X01,0);
    
            accumulator1 =vfmaq_n_f64(accumulator1, A01, vgetq_lane_f64(X11,0));
            
            A02 = vld1q_f64(valptr+2);
            Y12=vfmaq_laneq_f64(Y12, A02, X01,0);
            TEMP_1 = vfmaq_f64(TEMP_1, A02, X12); //
            TEMP_1 = vaddq_f64(TEMP_1, vextq_f64(TEMP_1, TEMP_1, 1));
            vst1_f64(Y2, vget_low_f64(TEMP_1));//ajouter TEMP_1 a Y2
    
            accumulator2 =vfmaq_n_f64(accumulator2, A02, vgetq_lane_f64(X11,0));
    
            TEMP_1 = {0,0};
            TEMP_1 =vzip2q_f64(Y21, TEMP_1);//Recuperer la valeur de Y21
            A11 = vld1q_f64(valptr+matsize);
            accumulator1 =vfmaq_n_f64(accumulator1, A11, vgetq_lane_f64(X11,1));
    
            TEMP_1 = vfmaq_f64(TEMP_1, A11, X11); //
            A12 = vld1q_f64(valptr+matsize+2);
            accumulator2 =vfmaq_n_f64(accumulator2, A12, vgetq_lane_f64(X11,1));

            Y12=vfmaq_laneq_f64(Y12, A12, X01,1);
            TEMP_1 = vfmaq_f64(TEMP_1, A12, X12); //
            TEMP_1 = vaddq_f64(TEMP_1, vextq_f64(TEMP_1, TEMP_1, 1));
            vst1q_f64(Y2+1, TEMP_1);
            TEMP_1 = {0,0};
            TEMP_1 =vzip1q_f64(Y22, TEMP_1);//Recuperer la valeur de Y21
            Y11=vfmaq_laneq_f64(Y11, A11, X01,1);
            A21 = vld1q_f64(valptr+2*matsize);
            accumulator1 =vfmaq_n_f64(accumulator1, A21, vgetq_lane_f64(X12,0));
            Y11=vfmaq_laneq_f64(Y11, A21, X02,0);
            TEMP_1 = vfmaq_f64(TEMP_1, A21, X11); //
            A22 = vld1q_f64(valptr+2*matsize+2);
            accumulator2 =vfmaq_n_f64(accumulator2, A22, vgetq_lane_f64(X12,0));
            TEMP_1 = vfmaq_f64(TEMP_1, A22, X12); //
            TEMP_1 = vaddq_f64(TEMP_1, vextq_f64(TEMP_1, TEMP_1, 1));
            vst1q_f64(Y2+2, TEMP_1);
            Y12 = vfmaq_laneq_f64(Y12, A22, X02,0);
            A31 = vld1q_f64(valptr+3*matsize);
    accumulator1 =vfmaq_n_f64(accumulator1, A31, vgetq_lane_f64(X12,1));
            Y11=vfmaq_laneq_f64(Y11, A31, X02,1);
            vst1q_f64(Y1, Y11);
            TEMP_1 = vmovq_n_f64(0);
            TEMP_1 =vzip2q_f64(Y22, TEMP_1);
            TEMP_1 = vfmaq_f64(TEMP_1, A31, X11); //s
            A32 = vld1q_f64(valptr+3*matsize+2);
    accumulator2 =vfmaq_n_f64(accumulator2, A32, vgetq_lane_f64(X12,1));
            TEMP_1 = vfmaq_f64(TEMP_1, A32, X12); //s
            TEMP_1 = vaddq_f64(TEMP_1, vextq_f64(TEMP_1, TEMP_1, 1));
           
    vst1_f64(Y2+3, vget_low_f64(TEMP_1));
            Y12=vfmaq_laneq_f64(Y12, A32, X02,1);
            vst1q_f64(Y1+2, Y12);
    vst1q_f64(Y1,accumulator1);
    vst1q_f64(Y1+2,accumulator2);
            
            
        
    }
void MatrixSymmetric::matrix_multiply_4x4_neonMid(const real* __restrict__ X1,real* __restrict__ Y1,int blocksize, int matsize, const real* __restrict__ valptr) const{
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

    float64x2_t Y11;
    float64x2_t Y12;
 
    X01 = vld1q_f64(X1);
    
    Y11 = vld1q_f64(Y1);
    Y12  = vld1q_f64(Y1+2);

    A01 = vld1q_f64(valptr);
    Y11=vfmaq_laneq_f64(Y11, A01, X01,0);
    
    A02 = vld1q_f64(valptr+2);
    Y12=vfmaq_laneq_f64(Y12, A02, X01,0);
    
    A11 = vld1q_f64(valptr+matsize);
    Y11=vfmaq_laneq_f64(Y11, A11, X01,1);
    
    A12 = vld1q_f64(valptr+matsize+2);
    Y12=vfmaq_laneq_f64(Y12, A12, X01,1);
    
    X02 = vld1q_f64(X1+2);
    A21 = vld1q_f64(valptr+2*matsize);
    Y11=vfmaq_laneq_f64(Y11, A21, X02,0);
    A22 = vld1q_f64(valptr+2*matsize+2);
    Y12=vfmaq_laneq_f64(Y12, A22, X02,0);
    
    A31 = vld1q_f64(valptr+3*matsize);
    Y11=vfmaq_laneq_f64(Y11, A31, X02,1);
    A32 = vld1q_f64(valptr+3*matsize+2);
    Y12=vfmaq_laneq_f64(Y12, A32, X02,1);

    vst1q_f64(Y1,Y11);
    vst1q_f64(Y1+2,Y12);
    
    

}


#if 0
void MatrixSymmetric::transVecMulAddBlock2(const real* X, real*Y1, real*Y2, int index_x1, int index_y1, int blocksize, int matsize) const
{

    real* ptr1 =(val + std::max((index_x1)*blocksize,index_y1 * blocksize) +dimension_ * std::min((index_x1)*blocksize,index_y1 * blocksize));
    real* ptr2 = (val + std::min((index_x1)*blocksize,index_y1 * blocksize) +dimension_ * std::max((index_x1)*blocksize,index_y1 * blocksize));
    const real* X1 = X+ index_y1*blocksize;
    
    for ( size_t i = 0; i < blocksize; ++i )
    {
        real value1 = 0;
        real value =0;
        for ( size_t j = 0; j <blocksize; ++j )
        {
            value1 += ptr1[j+i*matsize]*X1[j];
            Y2[j+blocksize*index_y1] += ptr1[j+i*matsize]* X1[i];
        }
        Y1[i+blocksize*index_x1] += value1;
        
    }
   
}
#endif
   


//------------------------------------------------------------------------------
void MatrixSymmetric::vecMulAdd( const real* X, real* Y ) const
{
    
    cblas_dsymv(CblasRowMajor, CblasLower, size_, 1.0, val, size_, X, 1, 1.0, Y, 1);
}


void MatrixSymmetric::vecMulAddIso2D( const real* X, real* Y ) const
{
    cblas_dsymv(CblasRowMajor, CblasLower, size_, 1.0, val, size_, X+0, 2, 1.0, Y+0, 2);
    cblas_dsymv(CblasRowMajor, CblasLower, size_, 1.0, val, size_, X+1, 2, 1.0, Y+1, 2);
}



void MatrixSymmetric::vecMulAddIso3D( const real* X, real* Y ) const
{
    
    cblas_dsymv(CblasRowMajor, CblasLower, size_, 1.0, val, size_, X+0, 3, 1.0, Y+0, 3);
    cblas_dsymv(CblasRowMajor,CblasLower, size_, 1.0, val, size_, X+1, 3, 1.0, Y+1, 3);
    cblas_dsymv(CblasRowMajor,CblasLower, size_, 1.0, val, size_, X+2, 3, 1.0, Y+2, 3);
}


void MatrixSymmetric::vecMulPerBlock( const real* X, real* Y,size_t blocksize, int nbThread ) const
{
    blockInstance* instance = new blockInstance(this, 4, X, Y, nbThread);
    //instance->testFullCalc(X,Y);
    instance->testFullCalcMt(X,Y);
    //instance->testFullMt(X, Y);
}
