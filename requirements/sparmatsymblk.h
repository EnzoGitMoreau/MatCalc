// Cytosim was created by Francois Nedelec.  Copyright 2020 Cambridge University.

#ifndef SPARMATSYMBLK_H
#define SPARMATSYMBLK_H
#define DIM 4
#define S_BLOCK_SIZE 4

#include <cstdio>
#include <iostream>
#include "assert_macro.h"
#include <deque>
#include <tuple>
#include "matrix44.h"


typedef std::tuple<int,int,int,Matrix44*> Tuple2;
typedef std::deque<Tuple2> Queue2;
struct mytuple
{
    Matrix44 matrix;
    int index_x;
    int index_y;

};

struct mytuplenew
{
    real a0;
    real a1;
    real a2;
    real a3;
    real a4;
    real a5;
    real a6;
    real a7;
    real a8;
    real a9;
    real aA;
    real aB;
    real aC;
    real aD;
    real aE;
    real aF;
    int index_x;
    int index_y;
};

struct mytuplenewtriple
{
    int nbMatAdded;
    
    int index_x;
    int index_y;
    int index1_x;
    int index1_y;
    int index2_x;
    int index2_y;
    real a0;
    real a1;
    real a2;
    real a3;
    real a4;
    real a5;
    real a6;
    real a7;
    real a8;
    real a9;
    real aA;
    real aB;
    real aC;
    real aD;
    real aE;
    real aF;
    
    real a10;
    real a11;
    real a12;
    real a13;
    real a14;
    real a15;
    real a16;
    real a17;
    real a18;
    real a19;
    real a1A;
    real a1B;
    real a1C;
    real a1D;
    real a1E;
    real a1F;
   
    real a20;
    real a21;
    real a22;
    real a23;
    real a24;
    real a25;
    real a26;
    real a27;
    real a28;
    real a29;
    real a2A;
    real a2B;
    real a2C;
    real a2D;
    real a2E;
    real a2F;
    
};

typedef struct mytuple mytuple;
typedef std::deque<mytuple> QueueT;
typedef std::deque<mytuplenew> QueueN;
typedef std::deque<mytuplenewtriple> QueueS;
/**
 The block size 'S_BLOCK_SIZE' can be defined on the command line during compilation,
 and is otherwise set here, to match the dimensionality of the simulation
 */


#if ( S_BLOCK_SIZE == 1 )
#   include "matrix11.h"
#elif ( S_BLOCK_SIZE == 2 )
#   include "matrix22.h"
#elif ( S_BLOCK_SIZE == 3 )
#   include "matrix33.h"
#elif ( S_BLOCK_SIZE == 4 )
#   include "matrix44.h"
#endif

///real symmetric sparse Matrix
/**
 The lower triangle of the matrix is stored.
 Elements are stored in no particular order in each column.

 SparMatSymBlk uses a sparse storage, with arrays of elements for each column.
 Each element is a full square block of size DIM x DIM.
 
 F. Nedelec, 17--27 March 2017, revised entirely June 2018
 */
class SparMatSymBlk final
{
public:

   
#if ( S_BLOCK_SIZE == 1 )
    typedef Matrix11 Block;
#elif ( S_BLOCK_SIZE == 2 )
    typedef Matrix22 Block;
#elif ( S_BLOCK_SIZE == 3 )
    typedef Matrix33 Block;
#elif ( S_BLOCK_SIZE == 4 )
    typedef Matrix44 Block;
#endif

    /// accessory class used to sort columns
    class Element;
    
    /// A column of the sparse matrix
    class Column
    {
        friend class SparMatSymBlk;
        friend class Meca;
    public:
        Block* blk_;    ///< all blocks
        unsigned* inx_; ///< line index for each element
        size_t alo_;    ///< allocated size of array
        size_t nbb_;    ///< number of blocks in column
        
        
        
        /// constructor
        Column();
        
        /// the assignment operator will transfer memory
        void operator = (Column&);
        
        /// destructor
        ~Column() { deallocate(); }
        
        /// allocate to hold 'nb' elements
        void allocate(size_t nb);
        
        /// deallocate memory
        void deallocate();
        
        /// set as zero
        void reset();
        
        /// sort element by increasing indices, using given temporary array
        void sortElements(Element[], size_t);
        
        /// print
        void printBlocks(std::ostream&) const;
        
        /// true if column is empty
        bool notEmpty() const { return ( nbb_ > 0 ); }
        
        /// return n-th block (not necessarily, located at line inx_[n]
        Block& operator[](size_t n) const { return blk_[n]; }
        
        /// return block corresponding to index
        Block* find_block(size_t j) const;
        
        /// return block located at line 'i' and column 'j'
        Block& block(size_t i, size_t j);
        
        Block& block2(size_t ii, size_t jj, Block* B);
        /// multiplication of a vector: Y <- Y + M * X, block_size = 1
        void vecMulAdd1D(const real* X, real* Y, size_t j) const;
        /// multiplication of a vector: Y <- Y + M * X, block_size = 2
        void vecMulAdd2D(const real* X, real* Y, size_t j) const;
        /// multiplication of a vector: Y <- Y + M * X, block_size = 3
        void vecMulAdd3D(const real* X, real* Y, size_t j) const;
        /// multiplication of a vector: Y <- Y + M * X, block_size = 4
        void vecMulAdd4D(const real* X, real* Y, size_t j) const;
        
        
        /// multiplication of a vector: Y <- Y + M * X with dim(X) = dim(M), block_size = 2
        void vecMulAdd2D_SSE(const double* X, double* Y, size_t j) const;
        /// multiplication of a vector: Y <- Y + M * X with dim(X) = dim(M), block_size = 2
        void vecMulAdd2D_AVX(const double* X, double* Y, size_t j) const;
        /// multiplication of a vector: Y <- Y + M * X with dim(X) = dim(M), block_size = 2
        void vecMulAdd2D_AVXU(const double* X, double* Y, size_t j) const;
        /// multiplication of a vector: Y <- Y + M * X with dim(X) = dim(M), block_size = 2
        void vecMulAdd2D_AVXUU(const double* X, double* Y, size_t j) const;
        /// multiplication of a vector: Y <- Y + M * X with dim(X) = dim(M), block_size = 3
        void vecMulAdd3D_SSE(const float* X, float* Y, size_t j) const;
        /// multiplication of a vector: Y <- Y + M * X with dim(X) = dim(M), block_size = 3
        void vecMulAdd3D_SSEU(const float* X, float* Y, size_t j) const;
        /// multiplication of a vector: Y <- Y + M * X with dim(X) = dim(M), block_size = 3
        void vecMulAdd3D_AVX(const double* X, double* Y, size_t j) const;
        /// multiplication of a vector: Y <- Y + M * X with dim(X) = dim(M), block_size = 3
        void vecMulAdd3D_AVXU(const double* X, double* Y, size_t j) const;
        /// multiplication of a vector: Y <- Y + M * X with dim(X) = dim(M), block_size = 4
        void vecMulAdd4D_AVX(const double* X, double* Y, size_t j) const;
        
        
        

    };

private:

    /// create Elements
    static size_t newElements(Element*& ptr, size_t);
    
    /// sort matrix block in increasing index order
    void sortElements();

public:
    
    /// size of matrix
    size_t rsize_;
    
    /// amount of memory which has been allocated
    size_t alloc_;

    /// array col_[c][] holds Elements of column 'c'
    Column * column_;
    
    /// colidx_[i] is the index of the first non-empty column of index >= i
    unsigned * colidx_;

public:
    
    /// return the size of the matrix
    size_t size() const { return rsize_ * S_BLOCK_SIZE; }
    
    /// change the size of the matrix
    void resize(size_t s) { rsize_ = s / S_BLOCK_SIZE; allocate(rsize_); }

    /// base for destructor
    void deallocate();
    
    /// default constructor
    SparMatSymBlk();
    
    /// default destructor
    ~SparMatSymBlk()  { deallocate(); }
    
    /// set to zero
    void reset();
    
    /// allocate the matrix to hold ( sz * sz )
    void allocate(size_t alc);
    
    /// number of columns
    size_t num_columns() const { return rsize_; }

    /// number of elements in j-th column
    size_t column_size(size_t j) const { assert_true(j<rsize_); return column_[j].nbb_; }
    
    /// line index of n-th element in j-th column (not multiplied by BLOCK_SIZE)
    size_t column_index(size_t j, size_t n) const { assert_true(j<rsize_); return column_[j].inx_[n]; }

    /// returns element at (i, i)
    Block& diag_block(size_t i);

    Block& block2(size_t ii, size_t jj, Block* B)
    {
        return column_[jj].block2(ii,jj,B);
    }
    /// returns element stored at line ii and column jj, if ( ii > jj )
    Block& block(const size_t ii, const size_t jj)
    {
        assert_true( ii < rsize_ );
#if ( 1 )
        assert_true( ii >= jj );
        return column_[jj].block(ii, jj);
#else
        assert_true( jj < rsize_ );
        size_t i = std::max(ii, jj);
        size_t j = std::min(ii, jj);
        return column_[j].block(i, j);
#endif
    }
    
    /// returns the address of element at line i, column j, no allocation is done
    real* addr(size_t i, size_t j) const;

    /// returns the address of element at line i, column j, allocating if necessary
    real& element(size_t i, size_t j);

    /// returns the address of element at line i, column j, allocating if necessary
    real& operator()(size_t i, size_t j) { return element(i,j); }

    /// scale the matrix by a scalar factor
    void scale(real);
    
    /// add terms with `i` and `j` in [start, start+cnt[ to `mat`
    void addDiagonalBlock(real* mat, size_t ldd, size_t start, size_t cnt, size_t mul) const;
    
    /// add scaled terms with `i` in [start, start+cnt[ if ( j > i ) and ( j <= i + rank ) to `mat`
    void addLowerBand(real alpha, real* mat, size_t ldd, size_t start, size_t cnt, size_t mul, size_t rank) const;

    /// add `alpha*trace()` for blocks within [start, start+cnt[ if ( j <= i + rank ) to `mat`
    void addDiagonalTrace(real alpha, real* mat, size_t ldd, size_t start, size_t cnt, size_t mul, size_t rank, bool sym) const;
    
    
    /// prepare matrix for multiplications by a vector (must be called)
    bool prepareForMultiply(int);

    /// multiplication of a vector, for columns within [start, stop[
    void vecMulAdd(const real*, real* Y, size_t start, size_t stop) const;
    /// multiplication of a vector: Y <- Y + M * X with dim(X) = dim(Y) = dim(M)
    void vecMulAdd(const real* X, real* Y) const { vecMulAdd(X, Y, 0, rsize_); }

    /// multiplication of a vector: Y <- Y + M * X with dim(X) = dim(Y) = dim(M)
    void vecMulAdd_ALT(const real* X, real* Y, size_t start, size_t stop) const;
    /// multiplication of a vector: Y <- Y + M * X with dim(X) = dim(Y) = dim(M)
    void vecMulAdd_ALT(const real* X, real* Y) const { vecMulAdd_ALT(X, Y, 0, rsize_); }
    
    /// 2D isotropic multiplication (not implemented)
    void vecMulAddIso2D(const real* X, real* Y) const {};
    /// 3D isotropic multiplication (not implemented)
    void vecMulAddIso3D(const real*, real*) const {};
    
    /// multiplication of a vector: Y <- Y + M * X with dim(X) = dim(M)
    void vecMul(const real* X, real* Y) const;

    /// true if matrix is non-zero
    bool notZero() const;
    
    /// number of blocks in columns [start, stop[. Set allocated size
    size_t nbElements(size_t start, size_t stop, size_t& alc) const;
    
    /// total number of blocks currently in use
    size_t nbElements() const { size_t alc=0; return nbElements(0, rsize_, alc); }

    /// returns a string which a description of the type of matrix
    std::string what() const;
    
    /// print matrix columns in sparse mode: ( i, j : value ) if |value| >= inf
    void printSparse(std::ostream&, real inf, size_t start, size_t stop) const;
    
    /// print matrix in sparse mode: ( i, j : value ) if |value| >= inf
    void printSparse(std::ostream& os, real inf) const { printSparse(os, inf, 0, rsize_); }

    /// print size of columns
    void printSummary(std::ostream&, size_t start, size_t stop);
    
    /// print
    void printBlocks(std::ostream&) const;

    /// debug function
    int bad() const;
    
    void vecMulMt(int nbThreads, const real* X, real* Y);
    void vecMulAddMt(int nbThreads, const real* X, real* Y);
    void calculate(std::deque<Queue2*>** workingPhases2, int phase_number, const real*X, real*Y, int big_mat_size);
    void testFullCalcMt(int nbThreads, const real* X, real* Y);
    void testFullCalcMtNtime(int nbThreads, const real* X, real* Y, int nTime);
    void testFullCalcMtNtime2(int nbThreads, const real* X, real* Y, int nTime);
    void testFullCalcMtNtime3(int nbThreads, const real* X, real* Y, int nTime);
    void work2(int nbThreads, const real* X, real* Y);
};


#endif

