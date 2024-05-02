//
//  blockMatrix.hpp
//  wtfmatrix
//
//  Created by Moreau Enzo on 03/04/2024.
//

#ifndef blockMatrix_hpp
#define blockMatrix_hpp
#include "real.h"
#include <cstdio>
#include <string>
#include "matsym.h"

#include <iostream>
class blockMatrix final
{
    private:
    size_t block_size;
    
    MatrixSymmetric* scdassociatedMatrix;

    public:

    
    const real* val;
    int position_x, position_y;

    size_t size() const{return block_size;};
    
    MatrixSymmetric* getMatrix() const{return scdassociatedMatrix;}

   
    blockMatrix(size_t bsize, int pos_x, int pos_y, const MatrixSymmetric* originalMatrix)
    {
        block_size = bsize;
        position_x = pos_x;
        scdassociatedMatrix = (MatrixSymmetric*) originalMatrix;
        position_y = pos_y;
    }
    blockMatrix(size_t bsize, int pos_x, int pos_y, const real* value)
    {
        block_size = bsize;
        position_x = pos_x;
        val = value;
        position_y = pos_y;
    }
    //Considering Y += XVec with X a block, considering the according positions
    void vecMult(const blockMatrix X, real* Vec,real* Y_res) const;
    void vecMultMiror(const blockMatrix X, real* Vec,real* Y_res) const;
    void calculateBlock(const blockMatrix X, real* Vec, real*Y1);
    void calculateBlockTest(const blockMatrix X, real* Vec, real*Y1, int matsize);
    void calculateBlockClass(const blockMatrix X, const real* Vec, real*Y1, int matsize)const;


};


#endif
