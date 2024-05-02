//
//  blockInstance.hpp
//  MatrixCalculation
//
//  Created by Moreau Enzo on 09/04/2024.
//

#ifndef blockInstance_hpp
#define blockInstance_hpp
//#define VERBOSE_2
#include <stdio.h>
#include <stdlib.h>
#include "matsym.h"
#include <iostream>       // std::cout
#include <deque>          // std::deque
#include <list>     
#include <ctime>
#include <tuple>     // std::list
#include <queue>
#include <stdexcept>
#include <boost/thread/barrier.hpp>
#include "blockingQueue.hpp"
#include "blockMatrix.hpp"
#include "boost/thread.hpp"
#include "arm_neon.h"
#include "boost/tuple/tuple.hpp"
#include "boost/lockfree/queue.hpp"
 /* blockInstance_hpp */

static inline void matrix_multiply_4x4_neond(const real* __restrict__ X1, const real* __restrict__ X2,real* __restrict__ Y1,  real* __restrict__ Y2, int blocksize, int matsize, const real* __restrict__ valptr) {
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
typedef std::tuple<int,int,int> Tuple;
typedef std::deque<Tuple> Queue;
class blockInstance final
{
private:
    const real* Vec;
public:
    
    size_t matsize;
    size_t blocksize;
    const MatrixSymmetric* matrix;
    const real* valptr;
    size_t block_per_line;
    size_t phase_number;
    int actualPhaseNb = 0;
    int threadsWaiting =0;
    int nbThreads = NB_THREADS;
    Queue* queue;
    boost::mutex _mutex;
    boost::mutex _waitingmutex;
    boost::mutex _producerWaitingMutex;
    real* Y;
    real* Y1;
    boost::condition_variable endPhaseCond;
    boost::condition_variable pleaseProduce;
    boost::condition_variable pleaseConsume;
    bool endPhase;
    int nbThreadFinishedPhase = 0;
    real* Y2;
    bool prodShouldWakeup = false;
    bool consShouldWakeup = false;
    bool shouldEnd = false;
    bool autorized_work = false;
    int ind=0;
    int t1;
    int worksize;
    int t2;
    int kunif=0;
    Queue***  workingPhases;
    
    int nbPhases;

    Queue** blockPhases;
    std::deque<Queue*>** workingPhases2;
    
    blockInstance(const MatrixSymmetric* matrixi,size_t block_size, const real*X, real*Yout,int nbThread)
    {
        //Construct the phases
        blocksize = block_size;
        matsize = matrixi->size();
        nbThreads = nbThread;
        block_per_line = matsize/blocksize;
        matrix = matrixi;
        valptr=matrixi->data();
        Vec = X;
        Y = Yout;
        real* Y_big =(real*) malloc(sizeof(real)*2*matsize);
        Y1 = Y_big;
        Y2 = Y_big + matsize;
        
        if(block_per_line%2 ==0)
        {
            phase_number = block_per_line/2 +1;
        }
        else{
            phase_number = (block_per_line +1)/2;
        }
        //Phases = cutIntoBlocks();
        queue = new Queue();
        
        
    }
    Queue** cutIntoBlocks()
    {
        Queue** blockPhases;
        
        blockPhases = (Queue**) malloc(phase_number *sizeof(Queue*));
        int stop = block_per_line;
        if(matsize % blocksize == 0)
        {
            for(int k=0; k<phase_number; k++)// for each phase create a queue
        {
            Queue* queuePhase= new Queue();
            int m = 0;
            
            if(k!=phase_number-1)
            {}
            else
            {
                if(block_per_line % 2 ==0)
                {
                    stop = block_per_line /2;
                }
            }
            for(int i=0; i<stop; i++)
            {
                int index_x = 0;
                int index_y = 0;
                if(k+i<stop)//maybe change
                {
                    index_x = k+i;
                    index_y = i;
                }
                else
                {
                    index_x = i;
                    index_y = (k+i)%block_per_line;
                }
                queuePhase->push_back(Tuple(index_x, index_y,1));
                m++;
                
            }
            
            
            blockPhases[k] = queuePhase;
        }
        }
        
        return blockPhases;
    }
    void fetchAndCalculate()
    {
        
        while(true)
        {
            
            
            //std::cout<<"thread in loop\n";
            int stop = block_per_line;
            bool last = false;
            _mutex.lock();
            
            if(!autorized_work)
            {
                
                _mutex.unlock();
                break;
            }
            _mutex.unlock();
            
            
            
            
            int actualPhase = actualPhaseNb;
            int i_s = ind;
            ind = (ind+1)%block_per_line;
            if(i_s != block_per_line -1)
            {_mutex.unlock();
            }
            else//Il faudra que les autres threads l'attendent
            {
                
            }
            //std::cout<<"\nPhase:"<<actualPhase;
            
            if(actualPhase != phase_number-1)
            {}
            else
            {
                if(block_per_line % 2 ==0)
                {
                    stop = block_per_line /2;
                }
            }
            if(i_s <stop)
            {
                int index_x = 0;
                int index_y = 0;
                if(actualPhase+i_s<stop)//maybe change
                {
                    index_x = actualPhase+i_s;
                    index_y = i_s;
                }
                else
                {
                    index_x = i_s;
                    index_y = (actualPhase+i_s)%block_per_line;
                }
                
                //std::cout<<"\nCalculation: ("<<index_x<<" , "<<index_y<<")\n";
                int t = std::max((index_x)*blocksize,index_y * blocksize) +matsize *std::min((index_x)*blocksize,index_y * blocksize);
                const real* valptr1 =(valptr +t);
                const real* X1 = this->Vec + (index_x * blocksize);
                const real* X2 = this->Vec + index_y * blocksize;
                real* Y1_ = Y1 + index_y * blocksize;
                real* Y2_  = Y2 + index_x * blocksize;
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
                    matrix->vecMulAddBlock2(Vec+(index_x* blocksize), Y1, index_x, index_y, blocksize,matsize);
                }
            }
            if(last)
            {
                actualPhaseNb++;
                
                std::cout<<"last\n";
                if( actualPhaseNb == phase_number)
                {
                    
                    autorized_work = false;
                }
                _mutex.unlock();
            }
            //std::cout<<"thread out loop\n";
        }
    }
    
    void testFullCalcMt2(const real*X, real*Y )
    {
        
        
        
        
        
        //   actualPhase = Phases[0];
        actualPhaseNb = 0;
        std::cout<<"Thread started\n";
        try
        {
            boost::thread* threads = (boost::thread*) malloc(sizeof(boost::thread)*nbThreads);
            threads[0] = boost::thread(boost::bind(&blockInstance::produce, this));
            for(int i=1;i<nbThreads; i++)
            {
                
                threads[i] = boost::thread(boost::bind(&blockInstance::consume2, this));;
            }
            
            std::cout<<"Thread started\n";
            
            for(int i=0;i<nbThreads; i++)
            {
                threads[i].join();
            }
        } catch (const std::exception &e) {
            std::cerr << "Exception occurred: " << e.what() << std::endl;
            // Handle exception here if needed
            
            
        }
        std::cout<<"Time between end and real end :" <<std::time(NULL) - t1;
        //std::cout<<"\nEvery threads have finished\n";
        for(int i=0;i<matsize;i++)
        {
            Y[i]+= Y1[i]+Y2[i];
        }
    }
    
    
    void produce()
    {
        //std::cout<<"\nEntering production "<<k<<"\n";
        //std::cout<<"\nPhase number is :"<<phase_number<<"\n";
        int k =0;
        while(k<phase_number)
        {
            //std::cout<<"\nEntering phase number : "<<k<<"\n Waiting for threads to start producing\n";
            
            
            
            //std::cout<<"Producer started producing phase number "<<k<<"\n";
            
            _mutex.lock();
            int stop = block_per_line;
            int m = 0;
            
            if(k!=phase_number-1)
            {}
            else
            {
                if(block_per_line % 2 ==0)
                {
                    stop = block_per_line /2;
                }
            }
            for(int i=0; i<stop; i++)
            {
                int index_x = 0;
                int index_y = 0;
                if(k+i<stop)//maybe change
                {
                    index_x = k+i;
                    index_y = i;
                }
                else
                {
                    index_x = i;
                    index_y = (k+i)%block_per_line;
                }
                queue->push_back(Tuple(index_x, index_y,1));
#ifdef VERBOSE_2
                std::cout<<"Producing : ("<<index_x<<" , "<<index_y<<")\n";
#endif
                
            }
            
            if(k == phase_number - 1)
            {
                shouldEnd = true;
            }
            worksize = queue->size();
            _mutex.unlock();
            //std::cout<<"Finished producing phase\n";
            
            
            
            //Add  a wait
            k++;
            prodShouldWakeup = false;
            consShouldWakeup = true;
            nbThreadFinishedPhase = 0;
            pleaseConsume.notify_all();
            //std::cout<<"notifying prod\n";
            
            if(k< phase_number)
            {
                boost::unique_lock<boost::mutex> lock(_producerWaitingMutex);
                
                while(!prodShouldWakeup)
                {
                    
                    pleaseProduce.wait(lock);
                    
                    
                }
                
                
            }
            
            
            
            
            
            
        }
        //Wait until the last phase is finished, to be able to tell consumers job is over
        
        //shouldEnd = true;
        std::cout<<"\nExit production \n";
        
        t1 = std::time(NULL);
        
        
        
        
    }
    
    void consume()
    {
        bool stop = false;
        bool globalStop = false;
        bool willhavetoEnd = false;
        Tuple* internalBuffer = (Tuple*) malloc(sizeof(Tuple) * block_per_line/(nbThreads -1));
        while(true)
        {
            
            
            boost::unique_lock<boost::mutex> lock(_waitingmutex);
            prodShouldWakeup = true;
            
            nbThreadFinishedPhase++;
            if(nbThreadFinishedPhase == nbThreads - 1 )
            {
                pleaseProduce.notify_all();
                
            }
            
            
            while(!consShouldWakeup)
            {
                pleaseConsume.wait(lock);
                
            }
            
            
            stop = false;
            while(!stop)
            {
                
                _mutex.lock();
                if(shouldEnd)
                {
                    willhavetoEnd = true;
                }
                if(queue->empty())
                {
                    _mutex.unlock();
                    stop = true;
                    break;
                    
                }
                else
                {
                    Tuple indexes = queue->front();
                    queue->pop_front();
                    
#ifdef VERBOSE_2
                    std::cout<<"\nConsuming : ("<<std::get<0>(indexes)<<", "<<std::get<1>(indexes)<<")\n";
#endif VERBOSE_2
                    _mutex.unlock();
                    int index_x =std::get<0>(indexes);
                    int index_y = std::get<1>(indexes);
                    int t = std::max((index_x)*blocksize,index_y * blocksize) +matsize *std::min((index_x)*blocksize,index_y * blocksize);
                    const real* valptr1 =(valptr +t);
                    const real* X1 = this->Vec + (index_x * blocksize);
                    const real* X2 = this->Vec + index_y * blocksize;
                    real* Y1_ = Y1 + index_y * blocksize;
                    real* Y2_  = Y2 + index_x * blocksize;
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
                        matrix->vecMulAddBlock2(Vec+(index_x* blocksize), Y1, index_x, index_y, blocksize,matsize);
                    }
                }
                
                
            }
            
            if(willhavetoEnd)
            {
                std::cout<<"end of consume thread\n";
                break;
            }
            
        }
        //std::cout<<"Out of here";
        
    }
    void consume2()
    {
        bool stop = false;
        bool globalStop = false;
        bool willhavetoEnd = false;
        Queue internalBuffer =  Queue();
        while(true)
        {
            
            
            
            
            _mutex.lock();
            if(shouldEnd)
            {
                //std::cout<<"will end";
                willhavetoEnd = true;
            }
            int a = 0;
            for(int m=0; m< block_per_line / (nbThreads - 1) +1; m++)
            {
                if(queue->empty())
                {
                    if(willhavetoEnd)
                    {
                        std::cout<<"breaking out";
                    }
                    break;
                }
                else
                {
                    Tuple indexes = queue->front();
                    queue->pop_front();
                    internalBuffer.push_front(indexes);
                }
                a++;
            }
            _mutex.unlock();
            
            int b=0;
            while(b<a)
            {
                
                if(willhavetoEnd && b ==0)
                {
                    std::cout<<"internal buff size"<<internalBuffer.size()<<"\n";
                }
                
#ifdef VERBOSE_2
                std::cout<<"\nConsuming : ("<<std::get<0>(indexes)<<", "<<std::get<1>(indexes)<<")\n";
#endif VERBOSE_2
                Tuple indexes = internalBuffer.front();
                internalBuffer.pop_front();
                int index_x =std::get<0>(indexes);
                int index_y = std::get<1>(indexes);
                int t = std::max((index_x)*blocksize,index_y * blocksize) +matsize *std::min((index_x)*blocksize,index_y * blocksize);
                const real* valptr1 =(valptr +t);
                const real* X1 = this->Vec + (index_x * blocksize);
                const real* X2 = this->Vec + index_y * blocksize;
                real* Y1_ = Y1 + index_y * blocksize;
                real* Y2_  = Y2 + index_x * blocksize;
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
                    matrix->vecMulAddBlock2(Vec+(index_x* blocksize), Y1, index_x, index_y, blocksize,matsize);
                }
                b++;
            }
            //std::cout<<"end calc\n";
            
            if(willhavetoEnd)
            {
                //std::cout<<"end of consume thread\n";
                break;
            }
            prodShouldWakeup = true;
            consShouldWakeup = false;
            //std::cout<<"Th nb "<<nbThreadFinishedPhase<<"finished phase\n";
            nbThreadFinishedPhase++;
            if(nbThreadFinishedPhase == nbThreads - 1 )
            {
                ///+std::cout<<"notifying ";
                
                pleaseProduce.notify_all();
                
            }
            
            boost::unique_lock<boost::mutex> lock(_waitingmutex);
            while(!consShouldWakeup)
            {
                pleaseConsume.wait(lock);
                
            }
            
            
            
            
        }
        //std::cout<<"Out of here";
        
    }
    
    //Working well
    
    void work(boost::barrier& barrier)
    {
        Queue* internalBuffer = new Queue();
        int k = 0;
        while(k<phase_number)
        {
            
            Queue* queue = blockPhases[k];
            //Data fetching
            _mutex.lock();
            for(int j=0; j< (block_per_line / (nbThreads) )+1; j++)
            {
                
                if(queue->empty())
                {
                    break;
                }
                else// We fetch and calculate next block
                {
                    internalBuffer->push_front(queue->front());
                    queue->pop_front();
                    
                }
            }
            _mutex.unlock();
            //std::cout<<"Boost thread nb :"<<boost::this_thread::get_id()<<"calculating phase nb"<<k<<"\n";
            //Calculation
            
            while(true)
            {
                    Tuple indexes;
                    if(internalBuffer->empty())
                    {
                        break;
                    }
                    else
                    {
                        indexes = internalBuffer->front();
                        internalBuffer->pop_front();
                        
                        int index_x =std::get<0>(indexes);
                        int index_y = std::get<1>(indexes);
                        int swap = std::get<2>(indexes);
#ifdef VERBOSE_2
                        std::cout<<"Consuming : "<<"("<<index_x<<","<<index_y<<","<<swap<<")"<<"\n";
#endif
                        int t = std::max((index_x)*blocksize,index_y * blocksize) +matsize *std::min((index_x)*blocksize,index_y * blocksize);
                        const real* valptr1 =(valptr +t);
                        const real* X1 = this->Vec + (index_x * blocksize);
                        const real* X2 = this->Vec + index_y * blocksize;
                        real* Y1_ = Y1 + index_y * blocksize;
                        real* Y2_  = Y2 + index_x * blocksize;
                        if(swap == 1)
                        {
                            const real* temp;
                            Y2_ = Y2+index_y * blocksize;
                            Y1_ = Y1+index_x * blocksize;
                            temp = X1;
                            X1 = X2;
                            X2 = temp;
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
                            matrix->vecMulAddBlock2(Vec+(index_x* blocksize), Y1, index_x, index_y, blocksize,matsize);
                        }
                    }
                }
            
            barrier.wait();
            //barrier.count_down_and_wait();
            k++;
           
        }
        
        
    }
    void testFullCalcMt(const real*X, real*Y)
    {
        
        std::cout<<"block_per_line:"<<block_per_line;
        std::cout<<"phase_number:"<<phase_number;
        blockPhases = (Queue**) malloc(phase_number *sizeof(Queue*));
        int stop = block_per_line;
        if(matsize % blocksize == 0)
        {
            for(int k=0; k<phase_number; k++)// for each phase create a queue
            {
                Queue* queuePhase= new Queue();
                int m = 0;
#ifdef VERBOSE_2
                std::cout<<"block_per_line:"<<block_per_line;
#endif
                if(k!=phase_number-1)
                {}
                else
                {
                    if(block_per_line % 2 ==0)
                    {
                        stop = block_per_line /2;
                    }
                }
                
                for(int i=0; i<stop; i++)
                {
                    int index_x = 0;
                    int index_y = 0;
                    int swap = 0;
                    if(k+i<stop)//maybe change
                    {
                        index_x = k+i;
                        index_y = i;
                    }
                    else
                    {
                        index_x = i;
                        index_y = (k+i)%block_per_line;
                        swap = 1;
                        
                    }
                    queuePhase->push_back(Tuple(index_x, index_y, swap));
#ifdef VERBOSE_2
                    std::cout<<"Producing : "<<"("<<index_x<<" , "<<index_y<<","<<swap<<")\n";
#endif
                    m++;
                    
                }
                
                
                blockPhases[k] = queuePhase;
            }
            
        }
        
        
        actualPhaseNb = 0;
       
        std::cout<<"Thread started\n";
        
        try
        {
            boost::thread* threads = (boost::thread*) malloc(sizeof(boost::thread)*nbThreads);
            boost::barrier bar(nbThreads);
            for(int i=0;i<nbThreads; i++)
            {
                threads[i] = boost::thread(boost::bind(&blockInstance::workTest, this,boost::ref(bar)));;
            }
            
            std::cout<<"Thread started\n";
            
            for(int i=0;i<nbThreads; i++)
            {
                threads[i].join();
            }
        } catch (const std::exception &e) {
            std::cerr << "Exception occurred: " << e.what() << std::endl;
            // Handle exception here if needed
            
            
        }
        std::cout<<"Time between end and real end :" <<std::time(NULL) - t1;
        //std::cout<<"\nEvery threads have finished\n";
        for(int i=0;i<matsize;i++)
        {
            Y[i]+= Y1[i]+Y2[i];
        }
    }
    
    void myCutintoblock()
    {
        blockPhases = (Queue**) malloc(phase_number *sizeof(Queue*));
        
        int stop = block_per_line;
        if(matsize % blocksize == 0)
        {
            for(int k=0; k<phase_number; k++)// for each phase create a queue
            {
                Queue* queuePhase= new Queue();
                int m = 0;
#ifdef VERBOSE_2
                std::cout<<"block_per_line:"<<block_per_line;
#endif
                if(k!=phase_number-1)
                {}
                else
                {
                    if(block_per_line % 2 ==0)
                    {
                        stop = block_per_line /2;
                    }
                }
                
                for(int i=0; i<stop; i++)
                {
                    int index_x = 0;
                    int index_y = 0;
                    int swap = 0;
                    if(k+i<stop)//maybe change
                    {
                        index_x = k+i;
                        index_y = i;
                    }
                    else
                    {
                        index_x = i;
                        index_y = (k+i)%block_per_line;
                        swap = 1;
                        
                    }
                    queuePhase->push_back(Tuple(index_x, index_y, swap));
#ifdef VERBOSE_2
                    std::cout<<"Producing : "<<"("<<index_x<<" , "<<index_y<<","<<swap<<")\n";
#endif
                    m++;
                    
                }
                
                
                blockPhases[k] = queuePhase;
             
            }
            
        }
    }
    
    void workTest(boost::barrier& barrier)
    {
        Queue* internalBuffer = new Queue();
        int k = 0;
        while(k<phase_number)
        {
            std::cout<<"Start of phase"<<k<<"\n";
            Queue* queue = blockPhases[k];
            //Data fetching
            
            while(true)
            {
                _mutex.lock();
                if(queue->empty())
                {
                    _mutex.unlock();
                    break;
                }
                for(int j=0; j< (block_per_line / (nbThreads * (nbThreads)) )+1; j++)
                {
                    
                    if(queue->empty())
                    {
                        break;
                    }
                    else// We fetch and calculate next block
                    {
                        internalBuffer->push_front(queue->front());
                        queue->pop_front();
                        
                    }
                }
                _mutex.unlock();
                //std::cout<<"Boost thread nb :"<<boost::this_thread::get_id()<<"calculating phase nb"<<k<<"\n";
                //Calculation
                
                while(true)
                {
                    Tuple indexes;
                    if(internalBuffer->empty())
                    {
                        break;
                    }
                    else
                    {
                        indexes = internalBuffer->front();
                        internalBuffer->pop_front();
                        
                        int index_x =std::get<0>(indexes);
                        int index_y = std::get<1>(indexes);
                        int swap = std::get<2>(indexes);
#ifdef VERBOSE_2
                        std::cout<<"Consuming : "<<"("<<index_x<<","<<index_y<<","<<swap<<")"<<"\n";
#endif
                        int t = std::max((index_x)*blocksize,index_y * blocksize) +matsize *std::min((index_x)*blocksize,index_y * blocksize);
                        const real* valptr1 =(valptr +t);
                        const real* X1 = this->Vec + (index_x * blocksize);
                        const real* X2 = this->Vec + index_y * blocksize;
                        real* Y1_ = Y1 + index_y * blocksize;
                        real* Y2_  = Y2 + index_x * blocksize;
                        if(swap == 1)
                        {
                            const real* temp;
                            Y2_ = Y2+index_y * blocksize;
                            Y1_ = Y1+index_x * blocksize;
                            temp = X1;
                            X1 = X2;
                            X2 = temp;
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
                            X1 = Vec+(index_x* blocksize);
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
            std::cout<<"End of phase"<<k<<"\n";
            barrier.wait();
            //barrier.count_down_and_wait();
            k++;
           
        }
        
    }
    
    
    
    //marche encore mieux
    
    void CalculateFromEquiPartPhase(const real*X, real*Y)
    {
        equiPartPhase();
        myCutintoblock();
        
        
        for(int k=0; k<nbPhases; k++)
        {
            //std::cout<<"Phase number : "<<k<<"\n";
            
            Queue** phase = workingPhases[k];
            for(int i =0; i<nbThreads; i++)
            {
                Queue* working_block = phase[i];
                //std::cout<<"Block number:"<<i<<"\n";
                while(!working_block->empty())
                {
                    Tuple indexes = working_block->front();
                    working_block->pop_front();
                    
                    
                    
                    int index_x =std::get<0>(indexes);
                    int index_y = std::get<1>(indexes);
                    int swap = std::get<2>(indexes);

                    //std::cout<<"Consuming : "<<"("<<index_x<<","<<index_y<<","<<swap<<")"<<"\n";

                    int t = std::max((index_x)*blocksize,index_y * blocksize) +matsize *std::min((index_x)*blocksize,index_y * blocksize);
                    const real* valptr1 =(valptr +t);
                    const real* X1 = this->Vec + (index_x * blocksize);
                    const real* X2 = this->Vec + index_y * blocksize;
                    real* Y1_ = Y1 + index_y * blocksize;
                    real* Y2_  = Y2 + index_x * blocksize;
                    if(swap == 1)
                    {
                        const real* temp;
                        Y2_ = Y2+index_y * blocksize;
                        Y1_ = Y1+index_x * blocksize;
                        temp = X1;
                        X1 = X2;
                        X2 = temp;
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
                        X1 = Vec+(index_x* blocksize);
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
        
        //std::cout<<"Usual phase list\n";
        for(int k=0; k<phase_number;k++)
        {
            //std::cout<<"Phase number"<<k<<"\n";
            
            Queue* phase = blockPhases[k];
            while(!phase->empty())
            {
                Tuple indexes = phase->front();
                phase->pop_front();
                int index_x =std::get<0>(indexes);
                int index_y = std::get<1>(indexes);
                int swap = std::get<2>(indexes);

                //std::cout<<"Consuming : "<<"("<<index_x<<","<<index_y<<","<<swap<<")"<<"\n";
                
            }
        }
        for(int i=0;i<matsize;i++)
        {
            Y[i]+= Y1[i]+Y2[i];
        }

    }
    
    
    void equiPartPhase()
    {
        //Here we want to construct phase that will give equirepartite work to each thread in the case of a full Matrix, this should be adapted considering the positions of the blocks in a SparMatSM
        
       workingPhases = (Queue***) malloc(nbThreads * sizeof(Queue**));
        
        workingPhases2 = (std::deque<Queue*>**) malloc(nbThreads * sizeof(std::deque<Queue*>*));
        int tBlock_size = matsize / (nbThreads*blocksize); //taille des subdivisions de travail
        
        nbPhases = nbThreads/2 + 1;
        int nbBlockInPhase = nbThreads;
        for(int k = 0; k<nbPhases; k++)
        {
            
            //How many blocks in a phase ?
            if(k == nbPhases - 1 && nbThreads%2 == 0)
            {
                nbBlockInPhase  = nbThreads/2;
            }
            Queue** phase = (Queue**) malloc(nbBlockInPhase*sizeof(Queue*));
            std::deque<Queue*>* phase_q = new std::deque<Queue*>();
            for(int m =0; m<nbBlockInPhase; m++)
            {
                Queue* working_block = new Queue();
                if(k==0)//First phase only contaisn half-blocks
                {
                    
                    for(int i = 0; i< tBlock_size; i++)
                    {
                        for(int j =0; j<=i;j++)
                        {
                            working_block->push_front(Tuple(i+m*tBlock_size,j+m*tBlock_size,0));
                            //Add(i+m*tBlock_size,j+m*t_Block_size);
                        }
                    }
                }
                else
                {
                    if(m<nbThreads - k)
                    {
                        for(int i=0; i<tBlock_size; i++)
                        {
                            for(int j=0; j<tBlock_size; j++)
                            {
                                //Add(k*tBlock_size + m*tBlock_size +i, m*tBlock_size +j)
                                working_block->push_front(Tuple(k*tBlock_size + m*tBlock_size +i, m*tBlock_size +j,0));
                            }
                        }
                    }
                    else
                    {
                        for(int i=0; i<tBlock_size; i++)
                        {
                            for(int j=0; j<tBlock_size; j++)
                            {
                                working_block->push_front(Tuple(m*tBlock_size +j, (k+m)%nbThreads*tBlock_size +i,1));
                                //Think about the swap
                                
                                //Add(k*tBlock_size + m*tBlock_size +i,)
                                //Add( m*tBlock_size +j, (k+m)%nbThreads*block_size +i)
                            }
                        }
                    }
                    
                }
                phase[m] = working_block;
                phase_q->push_front(working_block);
            }
            workingPhases[k] = phase;
            workingPhases2[k] = phase_q;
        }
        
        
    }
    
    void work2(boost::barrier& barrier)
    {
        Queue* internalBuffer;
        int k = 0;
        //std::cout<<"Nb phase is :"<<nbPhases<<"\n";
        while(k<nbPhases)
        {
            //std::cout<<k<<"\n";
            bool work = true;
            _mutex.lock();
            std::deque<Queue*>* phase = workingPhases2[k];
            if(phase->empty())
            {
                work = false;
            }
            else
            {
                internalBuffer = phase->front();
                phase->pop_front();
            }
            
            _mutex.unlock();
 
                while(work)
                {
                    Tuple indexes;
                    
                    if(internalBuffer->empty())
                    {
                        break;
                    }
                    else
                    {
                        indexes = internalBuffer->front();
                        internalBuffer->pop_front();
                        
                        int index_x =std::get<0>(indexes);
                        int index_y = std::get<1>(indexes);
                        int swap = std::get<2>(indexes);
#ifdef VERBOSE_2
                        std::cout<<"Consuming : "<<"("<<index_x<<","<<index_y<<","<<swap<<")"<<"\n";
#endif
                        int t = std::max((index_x)*blocksize,index_y * blocksize) +matsize *std::min((index_x)*blocksize,index_y * blocksize);
                        const real* valptr1 =(valptr +t);
                        const real* X1 = this->Vec + (index_x * blocksize);
                        const real* X2 = this->Vec + index_y * blocksize;
                        real* Y1_ = Y1 + index_y * blocksize;
                        real* Y2_  = Y2 + index_x * blocksize;
                        if(swap == 1)
                        {
                            const real* temp;
                            Y2_ = Y2+index_y * blocksize;
                            Y1_ = Y1+index_x * blocksize;
                            temp = X1;
                            X1 = X2;
                            X2 = temp;
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
                            X1 = Vec+(index_x* blocksize);
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
                
                
            
            barrier.wait();
            //barrier.count_down_and_wait();
            k++;
           
        }
    }
    
    void testFullMt(const real*X, real*Y)
    {
        
        equiPartPhase();
        try
        {
            boost::thread* threads = (boost::thread*) malloc(sizeof(boost::thread)*nbThreads);
            boost::barrier bar(nbThreads);
            for(int i=0;i<nbThreads; i++)
            {
                threads[i] = boost::thread(boost::bind(&blockInstance::work2, this,boost::ref(bar)));;
            }
            
            //std::cout<<"Thread started\n";
            
            for(int i=0;i<nbThreads; i++)
            {
                threads[i].join();
            }
        } catch (const std::exception &e) {
            std::cerr << "Exception occurred: " << e.what() << std::endl;
            // Handle exception here if needed
            
            
        }
        
        //std::cout<<"\nEvery threads have finished\n";
        for(int i=0;i<matsize;i++)
        {
            Y[i]+= Y1[i]+Y2[i];
        }
       
    }
    
};
    
        
       
        ///FINISH MY THREAD HERE
    
#endif /* blockInstance_h */

