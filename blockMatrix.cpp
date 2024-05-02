//
//  blockMatrix.cpp
//  wtfmatrix
//
//  Created by Moreau Enzo on 03/04/2024.
//

#include "blockMatrix.hpp"
//
void blockMatrix::vecMult(const blockMatrix X, real* Vec,real* Y_result) const
{
    scdassociatedMatrix->vecMulAddBlock2(Vec+(X.position_y * block_size), Y_result, X.position_x, X.position_y, block_size,scdassociatedMatrix->size());
    
   
}


void blockMatrix::calculateBlock(const blockMatrix X, real* Vec, real*Y1)
{
   
    int index_y= X.position_y;
    int index_x = X.position_x;
    int matsize = scdassociatedMatrix->size();
    const real* valptr =(scdassociatedMatrix->data() + std::max((X.position_x)*block_size,X.position_y * block_size) +scdassociatedMatrix->size() * std::min((X.position_x)*block_size,X.position_y* block_size));
    const real* X1 = Vec + index_x * block_size;
    const real* X2 = Vec + index_y * block_size;
    real* Y1_ = Y1 + index_y * block_size;
    real* Y2  = Y1 + index_x * block_size;
    scdassociatedMatrix->transVecMulAddBlock3(X1, X2, Y1_, Y2, (int)block_size, matsize,valptr);

    
   
}


void blockMatrix::calculateBlockTest(const blockMatrix X, real* Vec, real*Y1, int matsize)
{
   
        //scdassociatedMatrix->transVecMulAddBlock4(Vec, Y1, X.position_y, X.position_x,block_size, scdassociatedMatrix->size());
    int index_y= X.position_y;
    int index_x = X.position_x;
    const real* valptr =(scdassociatedMatrix->data() + std::max((X.position_x)*block_size,X.position_y * block_size) +matsize * std::min((X.position_x)*block_size,X.position_y* block_size));
    const real* X1 = Vec+index_y * block_size;
    const real* X2 = Vec + index_x*block_size;
    real* Y1_ = Y1 +index_x * block_size;
    real* Y2  = Y1+index_y * block_size;
    
    
    
    scdassociatedMatrix->matrix_multiply_4x4_neon2(X1, X2, Y1_, Y2, (int)block_size, matsize,valptr);

    
        
    
   
}
void blockMatrix::calculateBlockClass(const blockMatrix X, const real* Vec, real*Y1, int matsize)const
{
   
    int index_y= X.position_y;
    int index_x = X.position_x;
    const real* val = scdassociatedMatrix->data();
    const real* valptr =(val + std::max((X.position_x)*block_size,X.position_y * block_size) +matsize * std::min((X.position_x)*block_size,X.position_y* block_size));
    const real* X1 = Vec + index_x * block_size;
    const real* X2 = Vec + index_y * block_size;
    real* Y1_ = Y1 + index_y * block_size;
    real* Y2  = Y1 + index_x * block_size;
    if(index_x != index_y)
    {
        scdassociatedMatrix->transVecMulAddBlock3(X1, X2, Y1_, Y2, (int)block_size, matsize,valptr);
    }
    else
    {
        scdassociatedMatrix->vecMulAddBlock2(Vec+(X.position_y * block_size), Y1, index_x, index_y, block_size,matsize);
    }
   
}


