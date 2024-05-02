//
//  MatSymBMtInstance.hpp
//  MatrixCalculation
//
//  Created by Moreau Enzo on 17/04/2024.
//

#ifndef MatSymBMtInstance_hpp
#define MatSymBMtInstance_hpp

#include <stdio.h>
#include <arm_neon.h>
#include "sparmatsymblk.h"
#include "real.h"
#include "boost/thread.hpp"
#include <boost/thread/barrier.hpp>

class MatSymBMtInstance final
{
private:
    int phase_number;
    int* phase_tab;
    int nbThreads;
    boost::mutex _mutex;
    int big_mat_size;
    std::deque<Queue2*>** workingPhases2;
    real* Y1;
    int blocksize = 4;
    real* Y2;
    const real* X;
    SparMatSymBlk* matrix;
    int thNb =0;
    mytuplenew ***workingPhase4thway;
    mytuplenewtriple ***workingPhase5thway;
    real*** workingPhase6thway;
    Tuple2*** workingPhase2ndWay;
    int threadBlockSize;
    int*** workingIndexes;
    int** firstBlocks;
    unsigned short*** workingIndexes7;
    mytuple *** workingPhase3rdWay;
    int** work_lengths;
public:
    MatSymBMtInstance(SparMatSymBlk* matrix, int nbThreadsE)
    {
        nbThreads = nbThreadsE;
        this->matrix = matrix;
        big_mat_size = matrix->size();
        real* big_Y = (real*) malloc(big_mat_size*2*sizeof(real));
        Y1 = big_Y;
        Y2 = big_Y + big_mat_size;
        for(int i =0; i<big_mat_size;i++)
        {
            Y1[i] = 0;
            Y2[i] = 0;
        }
    }
    mytuplenewtriple getTupleThird(mytuplenew first, mytuplenew second, mytuplenew third)
    {
        mytuplenewtriple to_ret;
        to_ret.nbMatAdded = 3;
        
        to_ret.index_x = first.index_x;
        to_ret.index_y = first.index_y;
        to_ret.index1_x = second.index_x;
        to_ret.index1_y = second.index_y;
        to_ret.index2_x = third.index_x;
        to_ret.index2_y = third.index_y;
        to_ret.a0 = first.a0;
        to_ret.a1 = first.a1;
        to_ret.a2 = first.a2;
        to_ret.a3 = first.a3;
        to_ret.a4 = first.a4;
        to_ret.a5 = first.a5;
        to_ret.a6 = first.a6;
        to_ret.a7 = first.a7;
        to_ret.a8 = first.a8;
        to_ret.a9 = first.a9;
        to_ret.aA = first.aA;
        to_ret.aB = first.aB;
        to_ret.aC = first.aC;
        to_ret.aD = first.aD;
        to_ret.aE = first.aE;
        to_ret.aF = first.aF;
        
        to_ret.a10 = second.a0;
        to_ret.a11 = second.a1;
        to_ret.a12 = second.a2;
        to_ret.a13 = second.a3;
        to_ret.a14 = second.a4;
        to_ret.a15 = second.a5;
        to_ret.a16 = second.a6;
        to_ret.a17 = second.a7;
        to_ret.a18 = second.a8;
        to_ret.a19 = second.a9;
        to_ret.a1A = second.aA;
        to_ret.a1B = second.aB;
        to_ret.a1C = second.aC;
        to_ret.a1D = second.aD;
        to_ret.a1E = second.aE;
        to_ret.a1F = second.aF;
        
        to_ret.a20 = third.a0;
        to_ret.a21 = third.a1;
        to_ret.a22 = third.a2;
        to_ret.a23 = third.a3;
        to_ret.a24 = third.a4;
        to_ret.a25 = third.a5;
        to_ret.a26 = third.a6;
        to_ret.a27 = third.a7;
        to_ret.a28 = third.a8;
        to_ret.a29 = third.a9;
        to_ret.a2A = third.aA;
        to_ret.a2B = third.aB;
        to_ret.a2C = third.aC;
        to_ret.a2D = third.aD;
        to_ret.a2E = third.aE;
        to_ret.a2F = third.aF;
        
        return to_ret;
    }
    
    mytuplenewtriple getTupleThirdF2(mytuplenew first, mytuplenew second)
    {
        mytuplenewtriple to_ret;
        to_ret.nbMatAdded = 2;
        
        to_ret.index_x = first.index_x;
        to_ret.index_y = first.index_y;
        to_ret.index1_x = second.index_x;
        to_ret.index1_y = second.index_y;
        
        to_ret.a0 = first.a0;
        to_ret.a1 = first.a1;
        to_ret.a2 = first.a2;
        to_ret.a3 = first.a3;
        to_ret.a4 = first.a4;
        to_ret.a5 = first.a5;
        to_ret.a6 = first.a6;
        to_ret.a7 = first.a7;
        to_ret.a8 = first.a8;
        to_ret.a9 = first.a9;
        to_ret.aA = first.aA;
        to_ret.aB = first.aB;
        to_ret.aC = first.aC;
        to_ret.aD = first.aD;
        to_ret.aE = first.aE;
        to_ret.aF = first.aF;
        
        to_ret.a10 = second.a0;
        to_ret.a11 = second.a1;
        to_ret.a12 = second.a2;
        to_ret.a13 = second.a3;
        to_ret.a14 = second.a4;
        to_ret.a15 = second.a5;
        to_ret.a16 = second.a6;
        to_ret.a17 = second.a7;
        to_ret.a18 = second.a8;
        to_ret.a19 = second.a9;
        to_ret.a1A = second.aA;
        to_ret.a1B = second.aB;
        to_ret.a1C = second.aC;
        to_ret.a1D = second.aD;
        to_ret.a1E = second.aE;
        to_ret.a1F = second.aF;
        
        
        return to_ret;
    }
    
    mytuplenewtriple getTupleThirdF1(mytuplenew first)
    {
        mytuplenewtriple to_ret;
        to_ret.nbMatAdded = 1;
        
        to_ret.index_x = first.index_x;
        to_ret.index_y = first.index_y;
        
        to_ret.a0 = first.a0;
        to_ret.a1 = first.a1;
        to_ret.a2 = first.a2;
        to_ret.a3 = first.a3;
        to_ret.a4 = first.a4;
        to_ret.a5 = first.a5;
        to_ret.a6 = first.a6;
        to_ret.a7 = first.a7;
        to_ret.a8 = first.a8;
        to_ret.a9 = first.a9;
        to_ret.aA = first.aA;
        to_ret.aB = first.aB;
        to_ret.aC = first.aC;
        to_ret.aD = first.aD;
        to_ret.aE = first.aE;
        to_ret.aF = first.aF;
        
        
        return to_ret;
    }
    //Generate blocks in order to mltply
    void generateBlocks()
    {
        phase_tab = (int*) malloc(sizeof(int)*nbThreads);
        phase_tab[0] = 0;
        if(nbThreads %2 == 0)
        {
            phase_number = nbThreads /2 +1;
            for(int i =1; i<=nbThreads/2;i++)
            {
                phase_tab[i] = i;
                //std::cout << phase_tab[i] << " ";
            }
            for(int i=1 ; i<nbThreads/2;i++)
            {
                phase_tab[nbThreads/2 + i] = nbThreads/2 - i;
                //std::cout << phase_tab[nbThreads/2 + i] << " ";
            }
        }
        else
        {
            phase_number  = (nbThreads +1) / 2;
            for(int i =1; i<= (nbThreads - 1) /2; i++)
            {
                phase_tab[i] = 1;
                //std::cout << phase_tab[i] << " ";
            }
            for(int i=0; i< (nbThreads -1 )/ 2; i++)
            {
                phase_tab[(nbThreads-1)/2 +i] = (nbThreads-1)/2 -i;
                //std::cout << phase_tab[(nbThreads-1)/2  + i] << " ";
            }
        }
        
        workingPhases2 =(std::deque<Queue2*>**) malloc(sizeof(std::deque<Queue2*>*) * phase_number);
        Queue2*** phasesTemp = (Queue2***) malloc(sizeof(Queue2**) * phase_number);
        int threadBlockSize = matrix->size() / (S_BLOCK_SIZE * nbThreads);
        int* phase_counter = (int*) malloc(sizeof(int)*phase_number);
        
        //Init
        
        for(int i =0; i<phase_number; i++)
        {
            workingPhases2[i] = new std::deque<Queue2*>();
            phasesTemp[i] = (Queue2**) malloc(sizeof(Queue2*)*nbThreads);
            for(int j=0; j<nbThreads; j++)
            {
                phasesTemp[i][j] = new Queue2();
            }
        }
        
        int i =0;
        while(true)//Fetching blocks, assigned them to a phase each
        {
            int indX, indY;
            SparMatSymBlk::Column* col = &matrix->column_[i];
            if(col->nbb_>0)
            {
                //std::cout<<"Column number "<<matrix->colidx_[i]<<"->ind_YB : "<<(int) matrix->colidx_[i]/threadBlockSize<<"\n";
                indY = matrix->colidx_[i]/threadBlockSize;
                int indice_col = matrix->colidx_[i];
                for(int j=0; j<col->nbb_; j++)
                {
                    int indice_ligne = col->inx_[j];
                    //std::cout<<"Block position :"<<col->inx_[j]<<"-> ind_XB : "<<(int) col->inx_[j]/threadBlockSize<<"\n";
                    indX =col->inx_[j]/threadBlockSize;
                    //std::cout<<"Put in phase: "<<phase_tab[indX-indY]<<"\n";
                    int swap = 0;
                    if(indX - indY >= 1+ (int)(nbThreads/2))
                    {
                        swap =1;
                    }
                    //swap = 0; // Le swap ne fonctionne pas encore.
                    int phase =phase_tab[indX-indY];
                    if(phase>0)
                    {
                        if(swap ==1)
                        {
                            // std::cout<<"\nWARNINGINININING\n";
                        }
                    }
                    //phasesTemp[phase][phase_counter[phase]%nbThreads]->push_front(Tuple2( indice_ligne, indice_col,swap, &matrix->block(col->inx_[j],matrix->colidx_[i])));
                    if(swap==0)
                    {
                        phasesTemp[phase][indX]->push_front(Tuple2( indice_ligne, indice_col,swap, &col->blk_[j]));
                    }
                    else
                    {
                        phasesTemp[phase][indY]->push_front(Tuple2( indice_ligne, indice_col,swap, &col->blk_[j]));
                    }
                    
                    phase_counter[phase]++;
                    
                }
                //std::cout<<"\n\n";
            }
            
            i = matrix->colidx_[i+1];
            if(i>=matrix->rsize_)
            {
                
                break;
            }
            
        }
        
        for(int i=0; i<phase_number; i++)//Putting in order for threads
        {
            for(int j=0; j<nbThreads;j++)
            {
                workingPhases2[i]->push_front(phasesTemp[i][j]);
            }
        }
        
        
    }
    void generateBlocks2ndWay()
    {
        phase_tab = (int*) malloc(sizeof(int)*nbThreads);
        phase_tab[0] = 0;
        if(nbThreads %2 == 0)
        {
            phase_number = nbThreads /2 +1;
            for(int i =1; i<=nbThreads/2;i++)
            {
                phase_tab[i] = i;
                //std::cout << phase_tab[i] << " ";
            }
            for(int i=1 ; i<nbThreads/2;i++)
            {
                phase_tab[nbThreads/2 + i] = nbThreads/2 - i;
                //std::cout << phase_tab[nbThreads/2 + i] << " ";
            }
        }
        else
        {
            phase_number  = (nbThreads +1) / 2;
            for(int i =1; i<= (nbThreads - 1) /2; i++)
            {
                phase_tab[i] = 1;
                //std::cout << phase_tab[i] << " ";
            }
            for(int i=0; i< (nbThreads -1 )/ 2; i++)
            {
                phase_tab[(nbThreads-1)/2 +i] = (nbThreads-1)/2 -i;
                //std::cout << phase_tab[(nbThreads-1)/2  + i] << " ";
            }
        }
        
        workingPhase2ndWay = (Tuple2***) malloc(sizeof(Tuple2**)*phase_number);
        
        workingPhases2 =(std::deque<Queue2*>**) malloc(sizeof(std::deque<Queue2*>*) * phase_number);
        Queue2*** phasesTemp = (Queue2***) malloc(sizeof(Queue2**) * phase_number);
        int threadBlockSize = matrix->size() / (S_BLOCK_SIZE * nbThreads);
        int* phase_counter = (int*) malloc(sizeof(int)*phase_number);
        work_lengths = (int**) malloc(sizeof(int*)*phase_number);
        for(int i=0; i<phase_number;i++)
        {
            work_lengths[i] = (int*) malloc(sizeof(int)*nbThreads);
        }
        //Init
        
        for(int i =0; i<phase_number; i++)
        {
            workingPhase2ndWay[i] = (Tuple2**) malloc(sizeof(Tuple2*) * nbThreads);
            workingPhases2[i] = new std::deque<Queue2*>();
            phasesTemp[i] = (Queue2**) malloc(sizeof(Queue2*)*nbThreads);
            for(int j=0; j<nbThreads; j++)
            {
                phasesTemp[i][j] = new Queue2();
            }
        }
        
        int i =0;
        while(true)//Fetching blocks, assigned them to a phase each
        {
            int indX, indY;
            SparMatSymBlk::Column* col = &matrix->column_[i];
            if(col->nbb_>0)
            {
                //std::cout<<"Column number "<<matrix->colidx_[i]<<"->ind_YB : "<<(int) matrix->colidx_[i]/threadBlockSize<<"\n";
                indY = matrix->colidx_[i]/threadBlockSize;
                int indice_col = matrix->colidx_[i];
                for(int j=0; j<col->nbb_; j++)
                {
                    int indice_ligne = col->inx_[j];
                    //std::cout<<"Block position :"<<col->inx_[j]<<"-> ind_XB : "<<(int) col->inx_[j]/threadBlockSize<<"\n";
                    indX =col->inx_[j]/threadBlockSize;
                    //std::cout<<"Put in phase: "<<phase_tab[indX-indY]<<"\n";
                    int swap = 0;
                    if(indX - indY >= 1+ (int)(nbThreads/2))
                    {
                        swap =1;
                    }
                    //swap = 0; // Le swap ne fonctionne pas encore.
                    int phase =phase_tab[indX-indY];
                    if(phase>0)
                    {
                        if(swap ==1)
                        {
                            // std::cout<<"\nWARNINGINININING\n";
                        }
                    }
                    //phasesTemp[phase][phase_counter[phase]%nbThreads]->push_front(Tuple2( indice_ligne, indice_col,swap, &matrix->block(col->inx_[j],matrix->colidx_[i])));
                    if(swap==0)
                    {
                        phasesTemp[phase][indX]->push_front(Tuple2( indice_ligne, indice_col,swap, &col->blk_[j]));
                    }
                    else
                    {
                        phasesTemp[phase][indY]->push_front(Tuple2( indice_ligne, indice_col,swap, &col->blk_[j]));
                    }
                    
                    phase_counter[phase]++;
                    
                }
                //std::cout<<"\n\n";
            }
            
            i = matrix->colidx_[i+1];
            if(i>=matrix->rsize_)
            {
                
                break;
            }
            
        }
        
        for(int i=0; i<phase_number; i++)//Putting in order for threads
        {
            for(int j=0; j<nbThreads;j++)
            {
                workingPhases2[i]->push_front(phasesTemp[i][j]);
                work_lengths[i][j] = (int)phasesTemp[i][j]->size();
                workingPhase2ndWay[i][j] = (Tuple2*) malloc(sizeof(Tuple2)*work_lengths[i][j]);
                for(int m=0; m<work_lengths[i][j]; m++)
                {
                    workingPhase2ndWay[i][j][m] = phasesTemp[i][j]->front();
                    phasesTemp[i][j][0].pop_front();
                }
            }
        }
        
    }
    void generateBlocks3rdWay()
    {
        phase_tab = (int*) malloc(sizeof(int)*nbThreads);
        phase_tab[0] = 0;
        if(nbThreads %2 == 0)
        {
            phase_number = nbThreads /2 +1;
            for(int i =1; i<=nbThreads/2;i++)
            {
                phase_tab[i] = i;
                //std::cout << phase_tab[i] << " ";
            }
            for(int i=1 ; i<nbThreads/2;i++)
            {
                phase_tab[nbThreads/2 + i] = nbThreads/2 - i;
                //std::cout << phase_tab[nbThreads/2 + i] << " ";
            }
        }
        else
        {
            phase_number  = (nbThreads +1) / 2;
            for(int i =1; i<= (nbThreads - 1) /2; i++)
            {
                phase_tab[i] = 1;
                //std::cout << phase_tab[i] << " ";
            }
            for(int i=0; i< (nbThreads -1 )/ 2; i++)
            {
                phase_tab[(nbThreads-1)/2 +i] = (nbThreads-1)/2 -i;
                //std::cout << phase_tab[(nbThreads-1)/2  + i] << " ";
            }
        }
        
        workingPhase3rdWay = (mytuple***) malloc(sizeof(mytuple**)*phase_number);
        
        
        QueueT*** phasesTemp = (QueueT***) malloc(sizeof(QueueT**) * phase_number);
        int threadBlockSize = (int) matrix->size() / (S_BLOCK_SIZE * nbThreads);
        int* phase_counter = (int*) malloc(sizeof(int)*phase_number);
        work_lengths = (int**) malloc(sizeof(int*)*phase_number);
        for(int i=0; i<phase_number;i++)
        {
            work_lengths[i] = (int*) malloc(sizeof(int)*nbThreads);
        }
        //Init
        
        for(int i =0; i<phase_number; i++)
        {
            workingPhase3rdWay[i] = (mytuple**) malloc(sizeof(mytuple*) * nbThreads);
            
            phasesTemp[i] = (QueueT**) malloc(sizeof(QueueT*)*nbThreads);
            for(int j=0; j<nbThreads; j++)
            {
                phasesTemp[i][j] = new QueueT();
            }
        }
        
        int i =0;
        while(true)//Fetching blocks, assigned them to a phase each
        {
            int indX, indY;
            SparMatSymBlk::Column* col = &matrix->column_[i];
            if(col->nbb_>0)
            {
                //std::cout<<"Column number "<<matrix->colidx_[i]<<"->ind_YB : "<<(int) matrix->colidx_[i]/threadBlockSize<<"\n";
                indY = matrix->colidx_[i]/threadBlockSize;
                int indice_col = matrix->colidx_[i];
                for(int j=0; j<col->nbb_; j++)
                {
                    int indice_ligne = col->inx_[j];
                    //std::cout<<"Block position :"<<col->inx_[j]<<"-> ind_XB : "<<(int) col->inx_[j]/threadBlockSize<<"\n";
                    indX =col->inx_[j]/threadBlockSize;
                    //std::cout<<"Put in phase: "<<phase_tab[indX-indY]<<"\n";
                    int swap = 0;
                    if(indX - indY >= 1+ (int)(nbThreads/2))
                    {
                        swap =1;
                    }
                    //swap = 0; // Le swap ne fonctionne pas encore.
                    int phase =phase_tab[indX-indY];
                    if(phase>0)
                    {
                        if(swap ==1)
                        {
                            // std::cout<<"\nWARNINGINININING\n";
                        }
                    }
                    //phasesTemp[phase][phase_counter[phase]%nbThreads]->push_front(Tuple2( indice_ligne, indice_col,swap, &matrix->block(col->inx_[j],matrix->colidx_[i])));
                    mytuple data;
                    data.matrix = col->blk_[j];
                    data.index_x = indice_ligne * blocksize;
                    data.index_y = indice_col * blocksize;
                    
                    if(swap==0)
                    {
                        
                        phasesTemp[phase][indX]->push_front(data);
                    }
                    else
                    {
                        int temp = data.index_x;
                        data.index_x = data.index_y;
                        data.index_y = temp;
                        phasesTemp[phase][indY]->push_front(data);
                    }
                    
                    phase_counter[phase]++;
                    
                }
                //std::cout<<"\n\n";
            }
            
            i = matrix->colidx_[i+1];
            if(i>=matrix->rsize_)
            {
                
                break;
            }
            
        }
        
        for(int i=0; i<phase_number; i++)//Putting in order for threads
        {
            for(int j=0; j<nbThreads;j++)
            {
                
                work_lengths[i][j] = (int)phasesTemp[i][j]->size();
                workingPhase3rdWay[i][j] = (mytuple*) malloc(sizeof(mytuple)*work_lengths[i][j]);
                for(int m=0; m<work_lengths[i][j]; m++)
                {
                    workingPhase3rdWay[i][j][m] = phasesTemp[i][j]->front();
                    phasesTemp[i][j][0].pop_front();
                }
            }
        }
        
    }
    void generateBlocks4thWay()
    {
        phase_tab = (int*) malloc(sizeof(int)*nbThreads);
        phase_tab[0] = 0;
        if(nbThreads %2 == 0)
        {
            phase_number = nbThreads /2 +1;
            for(int i =1; i<=nbThreads/2;i++)
            {
                phase_tab[i] = i;
                //std::cout << phase_tab[i] << " ";
            }
            for(int i=1 ; i<nbThreads/2;i++)
            {
                phase_tab[nbThreads/2 + i] = nbThreads/2 - i;
                //std::cout << phase_tab[nbThreads/2 + i] << " ";
            }
        }
        else
        {
            phase_number  = (nbThreads +1) / 2;
            for(int i =1; i<= (nbThreads - 1) /2; i++)
            {
                phase_tab[i] = 1;
                //std::cout << phase_tab[i] << " ";
            }
            for(int i=0; i< (nbThreads -1 )/ 2; i++)
            {
                phase_tab[(nbThreads-1)/2 +i] = (nbThreads-1)/2 -i;
                //std::cout << phase_tab[(nbThreads-1)/2  + i] << " ";
            }
        }
        
        workingPhase4thway = (mytuplenew***) malloc(sizeof(mytuplenew**)*phase_number);
        
        
        QueueN*** phasesTemp = (QueueN***) malloc(sizeof(QueueN**) * phase_number);
        int threadBlockSize = (int) matrix->size() / (S_BLOCK_SIZE * nbThreads);
        int* phase_counter = (int*) malloc(sizeof(int)*phase_number);
        work_lengths = (int**) malloc(sizeof(int*)*phase_number);
        for(int i=0; i<phase_number;i++)
        {
            work_lengths[i] = (int*) malloc(sizeof(int)*nbThreads);
        }
        //Init
        
        for(int i =0; i<phase_number; i++)
        {
            workingPhase4thway[i] = (mytuplenew**) malloc(sizeof(mytuplenew*) * nbThreads);
            
            phasesTemp[i] = (QueueN**) malloc(sizeof(QueueN*)*nbThreads);
            for(int j=0; j<nbThreads; j++)
            {
                phasesTemp[i][j] = new QueueN();
            }
        }
        
        int i =0;
        while(true)//Fetching blocks, assigned them to a phase each
        {
            int indX, indY;
            SparMatSymBlk::Column* col = &matrix->column_[i];
            if(col->nbb_>0)
            {
                //std::cout<<"Column number "<<matrix->colidx_[i]<<"->ind_YB : "<<(int) matrix->colidx_[i]/threadBlockSize<<"\n";
                indY = matrix->colidx_[i]/threadBlockSize;
                int indice_col = matrix->colidx_[i];
                for(int j=0; j<col->nbb_; j++)
                {
                    int indice_ligne = col->inx_[j];
                    //std::cout<<"Block position :"<<col->inx_[j]<<"-> ind_XB : "<<(int) col->inx_[j]/threadBlockSize<<"\n";
                    indX =col->inx_[j]/threadBlockSize;
                    //std::cout<<"Put in phase: "<<phase_tab[indX-indY]<<"\n";
                    int swap = 0;
                    if(indX - indY >= 1+ (int)(nbThreads/2))
                    {
                        swap =1;
                    }
                    //swap = 0; // Le swap ne fonctionne pas encore.
                    int phase =phase_tab[indX-indY];
                    if(phase>0)
                    {
                        if(swap ==1)
                        {
                            // std::cout<<"\nWARNINGINININING\n";
                        }
                    }
                    //phasesTemp[phase][phase_counter[phase]%nbThreads]->push_front(Tuple2( indice_ligne, indice_col,swap, &matrix->block(col->inx_[j],matrix->colidx_[i])));
                    mytuplenew data;
                    Matrix44 matrix= col->blk_[j];
                    data.a0 = matrix.val[0x0];
                    data.a1 = matrix.val[0x1];
                    data.a2 = matrix.val[0x2];
                    data.a3 = matrix.val[0x3];
                    data.a4 = matrix.val[0x4];
                    data.a5 = matrix.val[0x5];
                    data.a6 = matrix.val[0x6];
                    data.a7 = matrix.val[0x7];
                    data.a8 = matrix.val[0x8];
                    data.a9 = matrix.val[0x9];
                    data.aA = matrix.val[0xA];
                    data.aB = matrix.val[0xB];
                    data.aC = matrix.val[0xC];
                    data.aD = matrix.val[0xD];
                    data.aE = matrix.val[0xE];
                    data.aF = matrix.val[0xF];
                    
                    data.index_x = indice_ligne * blocksize;
                    data.index_y = indice_col * blocksize;
                    
                    if(swap==0)
                    {
                        
                        phasesTemp[phase][indX]->push_front(data);
                    }
                    else
                    {
                        int temp = data.index_x;
                        data.index_x = data.index_y;
                        data.index_y = temp;
                        phasesTemp[phase][indY]->push_front(data);
                    }
                    
                    phase_counter[phase]++;
                    
                }
                //std::cout<<"\n\n";
            }
            
            i = matrix->colidx_[i+1];
            if(i>=matrix->rsize_)
            {
                
                break;
            }
            
        }
        
        for(int i=0; i<phase_number; i++)//Putting in order for threads
        {
            for(int j=0; j<nbThreads;j++)
            {
                
                work_lengths[i][j] = (int)phasesTemp[i][j]->size();
                workingPhase4thway[i][j] = (mytuplenew*) malloc(sizeof(mytuplenew)*work_lengths[i][j]);
                for(int m=0; m<work_lengths[i][j]; m++)
                {
                    workingPhase4thway[i][j][m] = phasesTemp[i][j]->front();
                    phasesTemp[i][j][0].pop_front();
                }
            }
        }
        
    }
    
    void generateBlocks5thWay()
    {
        phase_tab = (int*) malloc(sizeof(int)*nbThreads);
        phase_tab[0] = 0;
        if(nbThreads %2 == 0)
        {
            phase_number = nbThreads /2 +1;
            for(int i =1; i<=nbThreads/2;i++)
            {
                phase_tab[i] = i;
                //std::cout << phase_tab[i] << " ";
            }
            for(int i=1 ; i<nbThreads/2;i++)
            {
                phase_tab[nbThreads/2 + i] = nbThreads/2 - i;
                //std::cout << phase_tab[nbThreads/2 + i] << " ";
            }
        }
        else
        {
            phase_number  = (nbThreads +1) / 2;
            for(int i =1; i<= (nbThreads - 1) /2; i++)
            {
                phase_tab[i] = 1;
                //std::cout << phase_tab[i] << " ";
            }
            for(int i=0; i< (nbThreads -1 )/ 2; i++)
            {
                phase_tab[(nbThreads-1)/2 +i] = (nbThreads-1)/2 -i;
                //std::cout << phase_tab[(nbThreads-1)/2  + i] << " ";
            }
        }
        
        workingPhase5thway = (mytuplenewtriple***) malloc(sizeof(mytuplenew**)*phase_number);
        
        
        QueueN*** phasesTemp = (QueueN***) malloc(sizeof(QueueN**) * phase_number);
        int threadBlockSize = (int) matrix->size() / (S_BLOCK_SIZE * nbThreads);
        int* phase_counter = (int*) malloc(sizeof(int)*phase_number);
        work_lengths = (int**) malloc(sizeof(int*)*phase_number);
        for(int i=0; i<phase_number;i++)
        {
            work_lengths[i] = (int*) malloc(sizeof(int)*nbThreads);
        }
        //Init
        
        for(int i =0; i<phase_number; i++)
        {
            workingPhase5thway[i] = (mytuplenewtriple**) malloc(sizeof(mytuplenewtriple*) * nbThreads);
            
            phasesTemp[i] = (QueueN**) malloc(sizeof(QueueN*)*nbThreads);
            for(int j=0; j<nbThreads; j++)
            {
                phasesTemp[i][j] = new QueueN();
            }
        }
        
        int i =0;
        while(true)//Fetching blocks, assigned them to a phase each
        {
            int indX, indY;
            SparMatSymBlk::Column* col = &matrix->column_[i];
            if(col->nbb_>0)
            {
                //std::cout<<"Column number "<<matrix->colidx_[i]<<"->ind_YB : "<<(int) matrix->colidx_[i]/threadBlockSize<<"\n";
                indY = matrix->colidx_[i]/threadBlockSize;
                int indice_col = matrix->colidx_[i];
                
                for(int j=0; j<col->nbb_; j++)
                {
                    int indice_ligne = col->inx_[j];
                    //std::cout<<"Block position :"<<col->inx_[j]<<"-> ind_XB : "<<(int) col->inx_[j]/threadBlockSize<<"\n";
                    indX =col->inx_[j]/threadBlockSize;
                    //std::cout<<"Put in phase: "<<phase_tab[indX-indY]<<"\n";
                    int swap = 0;
                    if(indX - indY >= 1+ (int)(nbThreads/2))
                    {
                        swap =1;
                    }
                    //swap = 0; // Le swap ne fonctionne pas encore.
                    int phase =phase_tab[indX-indY];
                    if(phase>0)
                    {
                        if(swap ==1)
                        {
                            // std::cout<<"\nWARNINGINININING\n";
                        }
                    }
                    //phasesTemp[phase][phase_counter[phase]%nbThreads]->push_front(Tuple2( indice_ligne, indice_col,swap, &matrix->block(col->inx_[j],matrix->colidx_[i])));
                    mytuplenew data;
                    Matrix44 matrix= col->blk_[j];
                    data.a0 = matrix.val[0x0];
                    data.a1 = matrix.val[0x1];
                    data.a2 = matrix.val[0x2];
                    data.a3 = matrix.val[0x3];
                    data.a4 = matrix.val[0x4];
                    data.a5 = matrix.val[0x5];
                    data.a6 = matrix.val[0x6];
                    data.a7 = matrix.val[0x7];
                    data.a8 = matrix.val[0x8];
                    data.a9 = matrix.val[0x9];
                    data.aA = matrix.val[0xA];
                    data.aB = matrix.val[0xB];
                    data.aC = matrix.val[0xC];
                    data.aD = matrix.val[0xD];
                    data.aE = matrix.val[0xE];
                    data.aF = matrix.val[0xF];
                    
                    data.index_x = indice_ligne * blocksize;
                    data.index_y = indice_col * blocksize;
                    
                    if(swap==0)
                    {
                        
                        phasesTemp[phase][indX]->push_front(data);
                    }
                    else
                    {
                        int temp = data.index_x;
                        data.index_x = data.index_y;
                        data.index_y = temp;
                        phasesTemp[phase][indY]->push_front(data);
                    }
                    
                    phase_counter[phase]++;
                    
                }
                //std::cout<<"\n\n";
            }
            
            i = matrix->colidx_[i+1];
            if(i>=matrix->rsize_)
            {
                
                break;
            }
            
        }
        
        for(int i=0; i<phase_number; i++)//Putting in order for threads
        {
            for(int j=0; j<nbThreads;j++)
            {
                
                
                int t_work_length = (int)phasesTemp[i][j]->size();
                int nb_tuple = 0;
                if(t_work_length%3 ==0)
                {
                    nb_tuple = t_work_length/3;
                }
                else
                {
                    nb_tuple = t_work_length/3 +1;
                }
                work_lengths[i][j] = nb_tuple;
                
                workingPhase5thway[i][j] = (mytuplenewtriple*) malloc(sizeof(mytuplenewtriple)*nb_tuple);
                
                
                for(int m=0; m<(t_work_length/3); m++)
                {
                    mytuplenew first= phasesTemp[i][j]->front();
                    phasesTemp[i][j][0].pop_front();
                    mytuplenew second= phasesTemp[i][j]->front();
                    phasesTemp[i][j][0].pop_front();
                    mytuplenew third= phasesTemp[i][j]->front();
                    phasesTemp[i][j][0].pop_front();
                    workingPhase5thway[i][j][m] = getTupleThird(first, second, third);
                    
                }
                if(t_work_length%3==1)
                {
                    mytuplenew first= phasesTemp[i][j]->front();
                    phasesTemp[i][j][0].pop_front();
                    workingPhase5thway[i][j][(t_work_length/3)] = getTupleThirdF1(first);
                }
                else if(t_work_length%3==2)
                {
                    mytuplenew first= phasesTemp[i][j]->front();
                    phasesTemp[i][j][0].pop_front();
                    mytuplenew second= phasesTemp[i][j]->front();
                    phasesTemp[i][j][0].pop_front();
                    workingPhase5thway[i][j][(t_work_length/3)] = getTupleThirdF2(first, second);
                }
                else
                {
                    std::cout<<"wtf brox";
                }
            }
        }
    }
    
    void generateBlocks6thWay()
    {
        phase_tab = (int*) malloc(sizeof(int)*nbThreads);
        phase_tab[0] = 0;
        if(nbThreads %2 == 0)
        {
            phase_number = nbThreads /2 +1;
            for(int i =1; i<=nbThreads/2;i++)
            {
                phase_tab[i] = i;
                //std::cout << phase_tab[i] << " ";
            }
            for(int i=1 ; i<nbThreads/2;i++)
            {
                phase_tab[nbThreads/2 + i] = nbThreads/2 - i;
                //std::cout << phase_tab[nbThreads/2 + i] << " ";
            }
        }
        else
        {
            phase_number  = (nbThreads +1) / 2;
            for(int i =1; i<= (nbThreads - 1) /2; i++)
            {
                phase_tab[i] = 1;
                //std::cout << phase_tab[i] << " ";
            }
            for(int i=0; i< (nbThreads -1 )/ 2; i++)
            {
                phase_tab[(nbThreads-1)/2 +i] = (nbThreads-1)/2 -i;
                //std::cout << phase_tab[(nbThreads-1)/2  + i] << " ";
            }
        }
        
        workingPhase6thway = (real***) malloc(sizeof(real**)*phase_number);
        workingIndexes = (int***)malloc(sizeof(int**)*phase_number);
        
        QueueN*** phasesTemp = (QueueN***) malloc(sizeof(QueueN**) * phase_number);
        int threadBlockSize = (int) matrix->size() / (S_BLOCK_SIZE * nbThreads);
        int* phase_counter = (int*) malloc(sizeof(int)*phase_number);
        work_lengths = (int**) malloc(sizeof(int*)*phase_number);
        for(int i=0; i<phase_number;i++)
        {
            work_lengths[i] = (int*) malloc(sizeof(int)*nbThreads);
        }
        //Init
        
        for(int i =0; i<phase_number; i++)
        {
            workingPhase6thway[i] = (real**) malloc(sizeof(real*) * nbThreads);
            workingIndexes[i] = (int**) malloc(sizeof(int*) * nbThreads);
            
            phasesTemp[i] = (QueueN**) malloc(sizeof(QueueN*)*nbThreads);
            for(int j=0; j<nbThreads; j++)
            {
                phasesTemp[i][j] = new QueueN();
            }
        }
        
        int i =0;
        while(true)//Fetching blocks, assigned them to a phase each
        {
            int indX, indY;
            SparMatSymBlk::Column* col = &matrix->column_[i];
            if(col->nbb_>0)
            {
                //std::cout<<"Column number "<<matrix->colidx_[i]<<"->ind_YB : "<<(int) matrix->colidx_[i]/threadBlockSize<<"\n";
                indY = matrix->colidx_[i]/threadBlockSize;
                int indice_col = matrix->colidx_[i];
                
                for(int j=0; j<col->nbb_; j++)
                {
                    int indice_ligne = col->inx_[j];
                    //std::cout<<"Block position :"<<col->inx_[j]<<"-> ind_XB : "<<(int) col->inx_[j]/threadBlockSize<<"\n";
                    indX =col->inx_[j]/threadBlockSize;
                    //std::cout<<"Put in phase: "<<phase_tab[indX-indY]<<"\n";
                    int swap = 0;
                    if(indX - indY >= 1+ (int)(nbThreads/2))
                    {
                        swap =1;
                    }
                    //swap = 0; // Le swap ne fonctionne pas encore.
                    int phase =phase_tab[indX-indY];
                    if(phase>0)
                    {
                        if(swap ==1)
                        {
                            // std::cout<<"\nWARNINGINININING\n";
                        }
                    }
                    //phasesTemp[phase][phase_counter[phase]%nbThreads]->push_front(Tuple2( indice_ligne, indice_col,swap, &matrix->block(col->inx_[j],matrix->colidx_[i])));
                    mytuplenew data;
                    Matrix44 matrix= col->blk_[j];
                    data.a0 = matrix.val[0x0];
                    data.a1 = matrix.val[0x1];
                    data.a2 = matrix.val[0x2];
                    data.a3 = matrix.val[0x3];
                    data.a4 = matrix.val[0x4];
                    data.a5 = matrix.val[0x5];
                    data.a6 = matrix.val[0x6];
                    data.a7 = matrix.val[0x7];
                    data.a8 = matrix.val[0x8];
                    data.a9 = matrix.val[0x9];
                    data.aA = matrix.val[0xA];
                    data.aB = matrix.val[0xB];
                    data.aC = matrix.val[0xC];
                    data.aD = matrix.val[0xD];
                    data.aE = matrix.val[0xE];
                    data.aF = matrix.val[0xF];
                    
                    data.index_x = indice_ligne * blocksize;
                    data.index_y = indice_col * blocksize;
                    
                    if(swap==0)
                    {
                        
                        phasesTemp[phase][indX]->push_front(data);
                    }
                    else
                    {
                        int temp = data.index_x;
                        data.index_x = data.index_y;
                        data.index_y = temp;
                        phasesTemp[phase][indY]->push_front(data);
                    }
                    
                    phase_counter[phase]++;
                    
                }
                //std::cout<<"\n\n";
            }
            
            i = matrix->colidx_[i+1];
            if(i>=matrix->rsize_)
            {
                
                break;
            }
            
        }
        
        for(int i=0; i<phase_number; i++)//Putting in order for threads
        {
            for(int j=0; j<nbThreads;j++)
            {
                
                
                int t_work_length = (int)phasesTemp[i][j]->size();
                
                work_lengths[i][j] = t_work_length;
                
                workingPhase6thway[i][j] = (real*) malloc(sizeof(real)*t_work_length*16);
                workingIndexes[i][j] = (int*) malloc(sizeof(int)*t_work_length*2);
                
                
                for(int m=0; m<t_work_length; m++)
                {
                    
                   
                    mytuplenew first= phasesTemp[i][j]->front();
                    phasesTemp[i][j]->pop_front();
                    
                    workingPhase6thway[i][j][16*m] = first.a0;
                    workingPhase6thway[i][j][16*m+1] = first.a1;
                    workingPhase6thway[i][j][16*m+2] = first.a2;
                    workingPhase6thway[i][j][16*m+3] = first.a3;
                    workingPhase6thway[i][j][16*m+4] = first.a4;
                    workingPhase6thway[i][j][16*m+5] = first.a5;
                    workingPhase6thway[i][j][16*m+6] = first.a6;
                    workingPhase6thway[i][j][16*m+7] = first.a7;
                    workingPhase6thway[i][j][16*m+8] = first.a8;
                    workingPhase6thway[i][j][16*m+9] = first.a9;
                    workingPhase6thway[i][j][16*m+10] = first.aA;
                    workingPhase6thway[i][j][16*m+11] = first.aB;
                    workingPhase6thway[i][j][16*m+12] = first.aC;
                    workingPhase6thway[i][j][16*m+13] = first.aD;
                    workingPhase6thway[i][j][16*m+14] = first.aE;
                    workingPhase6thway[i][j][16*m+15] = first.aF;
                    workingIndexes[i][j][2*m] = first.index_x;
                    workingIndexes[i][j][2*m+1] = first.index_y;
                   // std::cout<<"Block sent : ("<<first.index_x<<", "<<first.index_y<<")"<<"\n";
                    
                    
                }
                
                
            }
        }
    }
    
    void calculate(const real* X, real* Y)
    {
        int big_mat_size = matrix->size();
        
        int blocksize = 4;
        int matsize= blocksize;
        
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
                    //std::cout<<"Calculating block : ("<<std::get<0>(bloc_to_calculate)<<","<<std::get<1>(bloc_to_calculate)<<","<<std::get<2>(bloc_to_calculate)<<") at adress: "<<std::get<3>(bloc_to_calculate)<<"\n";
                    
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
    
    void work2(boost::barrier& barrier)
    {
        Queue2* internalBuffer;
        int matsize = blocksize;
        int k = 0;
        while(k<phase_number)
        {
            //std::cout<<"\nStart of phase"<<k<<"\n";
            bool work = true;
            _mutex.lock();
            std::deque<Queue2*>* phase = workingPhases2[k];
            internalBuffer = new Queue2();
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
                Tuple2 bloc_to_calculate;
                
                if(internalBuffer->empty())
                {
                    break;
                }
                else
                {
                    bloc_to_calculate = internalBuffer->front();
                    internalBuffer->pop_front();
                    
                    int index_x =std::get<0>(bloc_to_calculate);
                    int index_y = std::get<1>(bloc_to_calculate);
                    int swap = std::get<2>(bloc_to_calculate);
                    real* valptr1 = std::get<3>(bloc_to_calculate)->val;
                    
                    
                    //std::cout<<"\nConsuming : "<<"("<<index_x<<","<<index_y<<","<<swap<<") at adress:"<<valptr1<<"\n";
                    
                    const real* X1 = X+ (index_y * blocksize);
                    const real* X2 = X + index_x * blocksize;
                    real* Y1_ = Y1 + index_x * blocksize;
                    real* Y2_  = Y2 + index_y * blocksize;
                    if(swap == 1)
                    {
                        
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
            
            //std::cout<<"\nEnd of phase"<<k<<"\n";
            barrier.wait();
            //barrier.count_down_and_wait();
            k++;
            
        }
    }
    
    void debugFunc()
    {
        int** matrice_debug = (int**) malloc(matrix->rsize_ * sizeof(int*));
        int** matrice_debug_tr = (int**) malloc(matrix->rsize_ * sizeof(int*));
        int** matrice_debug_sw = (int**) malloc(matrix->rsize_ * sizeof(int*));
        for(int i=0; i< matrix->rsize_; i++)
        {
            int*matrice_line = (int*) malloc(matrix->rsize_* sizeof(int));
            int*matrice_line_tr = (int*) malloc(matrix->rsize_* sizeof(int));
            int*matrice_line_sw = (int*) malloc(matrix->rsize_* sizeof(int));
            matrice_debug[i] = matrice_line;
            matrice_debug_tr[i] = matrice_line_tr;
            matrice_debug_sw[i] = matrice_line_sw;
        }
        
        
        for(int k =0; k<phase_number;k++)
        {
            std::deque<Queue2*>* phase = workingPhases2[k];
            int thread_nbr = 0;
            while(!phase->empty())
            {
                Queue2* internalBuffer = phase->front();
                phase->pop_front();
                while(!internalBuffer->empty())
                {
                    Tuple2 block = internalBuffer ->front();
                    internalBuffer ->pop_front();
                    int index_x = std::get<0>(block);
                    int index_y = std::get<1>(block);
                    int swap = std::get<2>(block);
                    int phase = k;
                    matrice_debug[index_x][index_y] = k+1;
                    matrice_debug_tr[index_x][index_y] =thread_nbr+1;
                    matrice_debug_sw[index_x][index_y] =swap+1;
                }
                thread_nbr ++;
            }
            
        }
        std::cout<<"Debug matrix\n";
        for(int i=0; i< matrix->rsize_; i++)
        {
            for(int j=0; j<matrix->rsize_;j++)
            {
                if(matrice_debug[i][j]>0)
                {
                    std::cout<<matrice_debug[i][j]-1<<" ";
                }
            }
            std::cout<<"\n";
        }
        std::cout<<"Debug matrix_tr_per_phase\n";
        for(int k=0; k<phase_number;k++)
        {
            std::cout<<"Phase :"<<k<<"\n";
            for(int i=0; i< matrix->rsize_; i++)
            {
                for(int j=0; j<matrix->rsize_;j++)
                {
                    if(matrice_debug_tr[i][j]>0 && matrice_debug[i][j] == k+1)
                    {
                        std::cout<<matrice_debug_tr[i][j]-1<<" ";
                    }
                }
                std::cout<<"\n";
            }
            std::cout<<"\n";
        }
        std::cout<<"Debug swap\n";
        for(int i=0; i< matrix->rsize_; i++)
        {
            for(int j=0; j<matrix->rsize_;j++)
            {
                if(matrice_debug_sw[i][j]>0)
                {
                    std::cout<<matrice_debug_sw[i][j]-1<<" ";
                }
            }
            std::cout<<"\n";
        }
    }
    void vecMulAdd(const real*X_calc, real*Y)
    {
        X= X_calc;
        generateBlocks();
        try
        {
            boost::thread* threads = (boost::thread*) malloc(sizeof(boost::thread)*nbThreads);
            boost::barrier bar(nbThreads);
            for(int i=0;i<nbThreads; i++)
            {
                threads[i] = boost::thread(boost::bind(&MatSymBMtInstance::work2, this,boost::ref(bar)));;
            }
            
            std::cout<<"Thread started\n";
            
            for(int i=0;i<nbThreads; i++)
            {
                threads[i].join();
            }
            for(int i=0;i<nbThreads; i++)
            {
                threads[i].interrupt();
            }
        }
        catch (const std::exception &e)
        {
            std::cerr << "Exception occurred: " << e.what() << std::endl;
            // Handle exception here if needed
            
        }
        
        for(int i=0;i<big_mat_size;i++)
        {
            
            Y[i]+= Y1[i]+Y2[i];
        }
        
        
    }
    int thread_number()
    {
        thNb++;
        return thNb-1;
    }
    void work2m(boost::barrier& barrier,boost::barrier& barrier2, int n_work)
    {
        _mutex.lock();
        int thread_nb = thread_number();
        //std::cout<<"Thread number:  "<<thread_nb<<"\n";
        _mutex.unlock();
        
        for(int m = 0; m<n_work;m++)
        {
            work5ntest(barrier2, thread_nb);
            barrier.wait();
        }
        
        
        
        
    }
    void work3m(boost::barrier& barrier,boost::barrier& barrier2, int n_work)
    {
        _mutex.lock();
        int thread_nb = thread_number();
        //std::cout<<"Thread number:  "<<thread_nb<<"\n";
        _mutex.unlock();
        
        for(int m = 0; m<n_work;m++)
        {
            work4ntest(barrier2, thread_nb);
            barrier.wait();
        }
        
        
        
        
    }
    void work4m(boost::barrier& barrier,boost::barrier& barrier2, int n_work)
    {
        _mutex.lock();
        int thread_nb = thread_number();
        //std::cout<<"Thread number:  "<<thread_nb<<"\n";
        _mutex.unlock();
        
        for(int m = 0; m<n_work;m++)
        {
            work5ntest(barrier2, thread_nb);
            barrier.wait();
        }
        
        
        
        
    }
    void work6m(boost::barrier& barrier,boost::barrier& barrier2, int n_work)
    {
        _mutex.lock();
        int thread_nb = thread_number();
        //std::cout<<"Thread number:  "<<thread_nb<<"\n";
        _mutex.unlock();
        
        for(int m = 0; m<n_work;m++)
        {
            work6ntest(barrier2, thread_nb);
            barrier.wait();
        }
        
        
        
        
    }
    
    void work7m(boost::barrier& barrier,boost::barrier& barrier2, int n_work)
    {
        _mutex.lock();
        int thread_nb = thread_number();
        //std::cout<<"Thread number:  "<<thread_nb<<"\n";
        _mutex.unlock();
        
        for(int m = 0; m<n_work;m++)
        {
            work7ntest(barrier2, thread_nb);
            barrier.wait();
        }
        
        
        
        
    }
    void work2n(boost::barrier& barrier2, int thNb)
    {
        
        int matsize = blocksize;
        int k = 0;
        Tuple2* work;
        int work_nb;
        Tuple2 bloc_to_calculate;
        int index_x;
        int index_y;
        int swap;
        real* valptr1;
        while(k<phase_number)
        {
            //std::cout<<"\nStart of phase"<<k<<"\n";
            
            _mutex.lock();
            work =  workingPhase2ndWay[k][thNb];
            work_nb = work_lengths[k][thNb];
            _mutex.unlock();
            
            for(int m=0; m<work_nb; m++)
            {
                bloc_to_calculate= work[m];
                index_x =std::get<0>(bloc_to_calculate);
                index_y = std::get<1>(bloc_to_calculate);
                swap = std::get<2>(bloc_to_calculate);
                valptr1 = std::get<3>(bloc_to_calculate)->val;
                
                
                //std::cout<<"\nConsuming : "<<"("<<index_x<<","<<index_y<<","<<swap<<") at adress:"<<valptr1<<"\n";
                
                const real* X1 = X+ (index_y * blocksize);
                const real* X2 = X + index_x * blocksize;
                real* Y1_ = Y1 + index_x * blocksize;
                real* Y2_  = Y2 + index_y * blocksize;
                if(swap == 1)
                {
                    
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
            
            
            barrier2.wait();
            //barrier.count_down_and_wait();
            k++;
            
        }
    }
    void work2ntest(boost::barrier& barrier2, int thNb)
    {
        
        int matsize = blocksize;
        int k = 0;
        Tuple2* work;
        int work_nb;
        Tuple2 bloc_to_calculate;
        int index_x;
        int index_y;
        int swap;
        real* valptr1;
        while(k<phase_number)
        {
            //std::cout<<"\nStart of phase"<<k<<"\n";
            
            _mutex.lock();
            work =  workingPhase2ndWay[k][thNb];
            work_nb = work_lengths[k][thNb];
            _mutex.unlock();
            
            for(int m=0; m<work_nb; m++)
            {
                auto & bloc_to_calculate= work[m];
                index_x =std::get<0>(bloc_to_calculate);
                index_y = std::get<1>(bloc_to_calculate);
                swap = std::get<2>(bloc_to_calculate);
                Matrix44* block = std::get<3>(bloc_to_calculate);
                valptr1 = std::get<3>(bloc_to_calculate)->val;
                
                
                //std::cout<<"\nConsuming : "<<"("<<index_x<<","<<index_y<<","<<swap<<") at adress:"<<valptr1<<"\n";
                
                const real* X1 = X+ (index_y * blocksize);
                const real* X2 = X + index_x * blocksize;
                real* Y1_ = Y1 + index_x * blocksize;
                real* Y2_  = Y2 + index_y * blocksize;
                if(swap == 1)
                {
                    
                    Y2_ = Y1 +index_y * blocksize; //Y2 "becomes" Y1
                    Y1_ = Y2 +index_x * blocksize;
                    
                    
                }
                if(index_x != index_y)
                {
                    //block->vecmul4_add(X1,Y1_);
                    block->bothvecmul(X1, X2, Y1_, Y2_);
                    //block->trans_vecmul4_add(X2,Y2_);
#if 0
                    for(int i=0;i<4;i++)
                    {
                        Y1_[i] += res1[i];
                        Y2_[i] += res2[i];
                    }
#endif
                }
                else
                {
                    Vector4 res1=block->vecmul(X1);
                    for(int i=0;i<4;i++)
                    {
                        Y1_[i] += res1[i];
                        
                    }
                }
            }
            
            
            barrier2.wait();
            //barrier.count_down_and_wait();
            k++;
            
        }
    }
    void work3ntest(boost::barrier& barrier2, int thNb)
    {
        
        int k = 0;
        mytuple* work;
        int work_nb;
        mytuple bloc_to_calculate;
        const real* X1;
        const real* X2;
        real* Y1_;
        real* Y2_;
        while(k<phase_number)
        {
            //std::cout<<"\nStart of phase"<<k<<"\n";
            
            _mutex.lock();
            work =  workingPhase3rdWay[k][thNb];
            work_nb = work_lengths[k][thNb];
            _mutex.unlock();
            
            for(mytuple* workp = work; workp < work + work_nb; workp++)
            {
                auto & bloc_to_calculate= *workp;
                
                int dist =bloc_to_calculate.index_x - bloc_to_calculate.index_y;
                
                if(dist >=0)
                {
                    X1 = X  + bloc_to_calculate.index_y;
                    X2 = X  + bloc_to_calculate.index_x;
                    Y1_= Y1 + bloc_to_calculate.index_x;
                    Y2_= Y2 + bloc_to_calculate.index_y;
                    
                }
                else
                {
                    
                    X1 = X  + bloc_to_calculate.index_x;
                    X2 = X  + bloc_to_calculate.index_y;
                    Y2_= Y1 + bloc_to_calculate.index_x; //Y2 "becomes" Y1
                    Y1_= Y2 + bloc_to_calculate.index_y;
                    
                    
                }
                
                if(dist !=0)
                {
                    
                    //bloc_to_calculate.matrix.bothvecmul(X1, X2, Y1_, Y2_);
                    real coeff;
                    real X1_val = X1[0];
                    real temp_res2 = 0;
                    real res1[4] = {0};
                    
                    auto &  val  = bloc_to_calculate.matrix.val;
                    coeff = val[0x0];
                    temp_res2 += coeff*X2[0];
                    res1[0x0] = coeff*X1_val;
                    coeff =val[0x1];
                    res1[0x1] += coeff*X1_val;
                    temp_res2 += coeff*X2[1];
                    coeff =val[0x2];
                    temp_res2 += coeff*X2[2];
                    res1[0x2] += coeff*X1_val;
                    coeff =val[0x3];
                    temp_res2 += coeff*X2[3];
                    res1[0x3] += coeff*X1_val;
                    coeff =val[0x4];
                    
                    Y2_[0x0]+= temp_res2;
                    
                    temp_res2 = 0;
                    X1_val = X1[1];
                    res1[0x0] += coeff*X1_val;
                    temp_res2 += coeff*X2[0];
                    
                    coeff =val[0x5];
                    temp_res2+= coeff*X2[1];
                    res1[0x1] += coeff*X1_val;
                    
                    coeff =val[0x6];
                    res1[0x2] += coeff*X1_val;
                    temp_res2 += coeff*X2[2];
                    
                    coeff =val[0x7];
                    res1[0x3] += coeff*X1_val;
                    temp_res2 += coeff*X2[3];
                    
                    coeff =val[0x8];
                    Y2_[0x1] +=temp_res2;
                    temp_res2 = 0;
                    X1_val = X1[2];
                    res1[0x0] += coeff*  X1_val;
                    temp_res2 += coeff*  X2[0];
                    
                    coeff =val[0x9];
                    res1[0x1] += coeff * X1_val;
                    temp_res2 += coeff * X2[1];
                    
                    coeff =val[0xA];
                    temp_res2 += coeff * X2[2];
                    res1[0x2] += coeff* X1_val;
                    
                    coeff =val[0xB];
                    temp_res2 += coeff * X2[3];
                    res1[0x3] += coeff * X1_val;
                    
                    coeff =val[0xC];
                    X1_val = X1[3];
                    Y2_[0x2]+= temp_res2;
                    temp_res2 = 0;
                    
                    temp_res2 += coeff* X2[0];
                    res1[0x0] += coeff* X1_val;
                    coeff =val[0xD];
                    
                    temp_res2 += coeff * X2[1];
                    res1[0x1] += coeff * X1_val;
                    coeff =val[0xE];
                    
                    temp_res2 += coeff * X2[2];
                    res1[0x2] += coeff * X1_val;
                    
                    coeff =val[0xF];
                    temp_res2 += coeff* X2[3];
                    
                    res1[0x3] += coeff * X1_val;
                    
                    Y2_[0x3] += temp_res2;
                    
                    for(int i =0; i<4;i++)
                    {
                        Y1_[i]+= res1[i];
                    }
                    
                    
                }
                else
                {
                    Vector4 res1=bloc_to_calculate.matrix.vecmul(X1);
                    for(int i=0;i<4;i++)
                    {
                        Y1_[i] += res1[i];
                        
                    }
                }
                
            }
            
            barrier2.wait();
            k++;
            
        }
    }
    void work4ntest(boost::barrier& barrier2, int thNb)
    {
        
        int k = 0;
        mytuple* work;
        int work_nb;
        mytuple bloc_to_calculate;
        const real* X1;
        const real* X2;
        real* Y1_;
        real* Y2_;
        while(k<phase_number)
        {
            //std::cout<<"\nStart of phase"<<k<<"\n";
            
            _mutex.lock();
            work =  workingPhase3rdWay[k][thNb];
            work_nb = work_lengths[k][thNb];
            _mutex.unlock();
            
            for(mytuple* workp = work; workp < work + work_nb; workp++)
            {
                auto & bloc_to_calculate= *workp;
                
                int dist =bloc_to_calculate.index_x - bloc_to_calculate.index_y;
                
                if(dist >=0)
                {
                    X1 = X  + bloc_to_calculate.index_y;
                    X2 = X  + bloc_to_calculate.index_x;
                    Y1_= Y1 + bloc_to_calculate.index_x;
                    Y2_= Y2 + bloc_to_calculate.index_y;
                    
                }
                else
                {
                    
                    X1 = X  + bloc_to_calculate.index_x;
                    X2 = X  + bloc_to_calculate.index_y;
                    Y2_= Y1 + bloc_to_calculate.index_x; //Y2 "becomes" Y1
                    Y1_= Y2 + bloc_to_calculate.index_y;
                    
                    
                }
                
                if(dist !=0)
                {
                    auto &  val  = bloc_to_calculate.matrix.val;
                    real y10 = 0;
                    real y11 = 0;
                    real y12=0;
                    real y13=0;
                    real y20 = 0;
                    real y21 = 0;
                    real y22 = 0;
                    real y23=0;
                    real r10 = X1[0];
                    real r11 = X1[1];
                    real r12 = X1[2];
                    real r13 = X1[3];
                    real r20 = X2[0];
                    real r21 = X2[1];
                    real r22 = X2[2];
                    real r23 = X2[3];
                    {real & c = val[0x0];
                        
                        y10 += c*r10;
                        y20 += c*r20;
                    }
                    {
                        real & c = val[0x1];
                        y11 += c * r10;
                        y20 += c * r21;
                    }
                    {
                        real & c = val[0x4];
                        y10 += c * r11;
                        y21 += c * r20;
                    }
                    {
                        real & c = val[0x2];
                        y12 += c * r10;
                        y20 += c * r22;
                    }
                    {
                        real& c = val[0x8];
                        y10 += c * r12;
                        y22 += c* r20;
                    }
                    {
                        real & c = val[0x3];
                        y13 += c * r10;
                        y20 += c * r23;
                    }
                    {
                        real & c  = val[0xC];
                        y10 += c *r13;
                        y23 += c * r20;
                    }
                    {
                        real & c = val[0x5];
                        y11+= c*r11;
                        y21 += c *r21;
                    }
                    {
                        real & c=  val[0x6];
                        y12 += c*r11;
                        y21 += c*r22;
                    }
                    {
                        real & c = val[0x9];
                        y11 += c *r12;
                        y22 += c *r21;
                    }
                    {
                        real &c  = val[0x7];
                        y13 += c *r11;
                        y21 += c * r23;
                    }
                    {
                        real & c  = val[0xD];
                        y11 +=c*r13;
                        y23 += c *r21;
                    }
                    {
                        real & c  = val[0xA];
                        y12 += c * r12;
                        y22 += c* r22;
                    }
                    {
                        real & c = val[0xE];
                        y12 += c * r13;
                        y23 += c * r22;
                    }
                    {
                        real & c = val[0xB];
                        y13 += c * r12;
                        y22 += c *r23;
                    }
                    {
                        real & c = val[0xF];
                        y13 += c * r13;
                        y23 += c *r23;
                    }
                    
                    Y1_[0] += y10;
                    Y1_[1] += y11;
                    Y1_[2] += y12;
                    Y1_[3] += y13;
                    Y2_[0] += y20;
                    Y2_[1] += y21;
                    Y2_[2] += y22;
                    Y2_[3] += y23;
                    
                    
                    
                }
                else
                {
                    Vector4 res1=bloc_to_calculate.matrix.vecmul(X1);
                    for(int i=0;i<4;i++)
                    {
                        Y1_[i] += res1[i];
                        
                    }
                }
                
            }
            
            barrier2.wait();
            k++;
            
        }
    }
    void work5ntest(boost::barrier& barrier2, int thNb)
    {
        
        
        int k = 0;
        mytuplenewtriple* work;
        int work_nb;
        
        const real* X1;
        const real* X2;
        real* Y1_;
        real* Y2_;
        while(k<phase_number)
        {
            //std::cout<<"\nStart of phase"<<k<<"\n";
            
            _mutex.lock();
            work =  workingPhase5thway[k][thNb];
            work_nb = work_lengths[k][thNb];
            _mutex.unlock();
            
            for(mytuplenewtriple* workp = work; workp < work + work_nb; workp++)
            {
                
                
                auto & bloc_to_calculate= *workp;
                
                int dist =bloc_to_calculate.index_x - bloc_to_calculate.index_y;
                real a0 = bloc_to_calculate.a0;
                real a1 = bloc_to_calculate.a1;
                real a2 = bloc_to_calculate.a2;
                real a3 = bloc_to_calculate.a3;
                real a4 = bloc_to_calculate.a4;
                real a5 = bloc_to_calculate.a5;
                real a6 = bloc_to_calculate.a6;
                real a7 = bloc_to_calculate.a7;
                real a8 = bloc_to_calculate.a8;
                real a9 = bloc_to_calculate.a9;
                real aA = bloc_to_calculate.aA;
                real aB = bloc_to_calculate.aB;
                real aC = bloc_to_calculate.aC;
                real aD = bloc_to_calculate.aD;
                real aE = bloc_to_calculate.aE;
                real aF = bloc_to_calculate.aF;
                
                if(dist >=0)
                {
                    X1 = X  + bloc_to_calculate.index_y;
                    X2 = X  + bloc_to_calculate.index_x;
                    Y1_= Y1 + bloc_to_calculate.index_x;
                    Y2_= Y2 + bloc_to_calculate.index_y;
                    
                }
                else
                {
                    
                    X1 = X  + bloc_to_calculate.index_x;
                    X2 = X  + bloc_to_calculate.index_y;
                    Y2_= Y1 + bloc_to_calculate.index_x; //Y2 "becomes" Y1
                    Y1_= Y2 + bloc_to_calculate.index_y;
                    
                    
                }
                
                if(dist !=0)
                {
                    
                    real y10 = 0;
                    real y11 = 0;
                    real y12=0;
                    real y13=0;
                    real y20 = 0;
                    real y21 = 0;
                    real y22 = 0;
                    real y23=0;
                    real r10 = X1[0];
                    real r11 = X1[1];
                    real r12 = X1[2];
                    real r13 = X1[3];
                    real r20 = X2[0];
                    real r21 = X2[1];
                    real r22 = X2[2];
                    real r23 = X2[3];
                    {real & c = a0;
                        
                        y10 += c*r10;
                        y20 += c*r20;
                    }
                    {
                        real & c = a1;
                        y11 += c * r10;
                        y20 += c * r21;
                    }
                    {
                        real & c = a4;
                        y10 += c * r11;
                        y21 += c * r20;
                    }
                    {
                        real & c = a2;
                        y12 += c * r10;
                        y20 += c * r22;
                    }
                    {
                        real& c = a8;
                        y10 += c * r12;
                        y22 += c* r20;
                    }
                    {
                        real & c = a3;
                        y13 += c * r10;
                        y20 += c * r23;
                    }
                    {
                        real & c  = aC;
                        y10 += c *r13;
                        y23 += c * r20;
                    }
                    {
                        real & c = a5;
                        y11+= c*r11;
                        y21 += c *r21;
                    }
                    {
                        real & c=  a6;
                        y12 += c*r11;
                        y21 += c*r22;
                    }
                    {
                        real & c = a9;
                        y11 += c *r12;
                        y22 += c *r21;
                    }
                    {
                        real &c  = a7;
                        y13 += c *r11;
                        y21 += c * r23;
                    }
                    {
                        real & c  = aD;
                        y11 +=c*r13;
                        y23 += c *r21;
                    }
                    {
                        real & c  = aA;
                        y12 += c * r12;
                        y22 += c* r22;
                    }
                    {
                        real & c = aE;
                        y12 += c * r13;
                        y23 += c * r22;
                    }
                    {
                        real & c = aB;
                        y13 += c * r12;
                        y22 += c *r23;
                    }
                    {
                        real & c = aF;
                        y13 += c * r13;
                        y23 += c *r23;
                    }
                    
                    Y1_[0] += y10;
                    Y1_[1] += y11;
                    Y1_[2] += y12;
                    Y1_[3] += y13;
                    Y2_[0] += y20;
                    Y2_[1] += y21;
                    Y2_[2] += y22;
                    Y2_[3] += y23;
                    
                    
                    
                }
                else
                {
                    Matrix44* restored = new Matrix44(a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,aA,aB,aC,aD,aE,aF);
                    Vector4 res1=restored->vecmul(X1);
                    for(int i=0;i<4;i++)
                    {
                        Y1_[i] += res1[i];
                        
                    }
                }
                
                if(bloc_to_calculate.nbMatAdded >=2)
                {
                    int dist =bloc_to_calculate.index1_x - bloc_to_calculate.index1_y;
                    real a0 = bloc_to_calculate.a10;
                    real a1 = bloc_to_calculate.a11;
                    real a2 = bloc_to_calculate.a12;
                    real a3 = bloc_to_calculate.a13;
                    real a4 = bloc_to_calculate.a14;
                    real a5 = bloc_to_calculate.a15;
                    real a6 = bloc_to_calculate.a16;
                    real a7 = bloc_to_calculate.a17;
                    real a8 = bloc_to_calculate.a18;
                    real a9 = bloc_to_calculate.a19;
                    real aA = bloc_to_calculate.a1A;
                    real aB = bloc_to_calculate.a1B;
                    real aC = bloc_to_calculate.a1C;
                    real aD = bloc_to_calculate.a1D;
                    real aE = bloc_to_calculate.a1E;
                    real aF = bloc_to_calculate.a1F;
                    
                    
                    if(dist >=0)
                    {
                        X1 = X  + bloc_to_calculate.index1_y;
                        X2 = X  + bloc_to_calculate.index1_x;
                        Y1_= Y1 + bloc_to_calculate.index1_x;
                        Y2_= Y2 + bloc_to_calculate.index1_y;
                        
                    }
                    else
                    {
                        
                        X1 = X  + bloc_to_calculate.index1_x;
                        X2 = X  + bloc_to_calculate.index1_y;
                        Y2_= Y1 + bloc_to_calculate.index1_x; //Y2 "becomes" Y1
                        Y1_= Y2 + bloc_to_calculate.index1_y;
                        
                        
                    }
                    
                    if(dist !=0)
                    {
                        
                        real y10 = 0;
                        real y11 = 0;
                        real y12=0;
                        real y13=0;
                        real y20 = 0;
                        real y21 = 0;
                        real y22 = 0;
                        real y23=0;
                        real r10 = X1[0];
                        real r11 = X1[1];
                        real r12 = X1[2];
                        real r13 = X1[3];
                        real r20 = X2[0];
                        real r21 = X2[1];
                        real r22 = X2[2];
                        real r23 = X2[3];
                        {real & c = a0;
                            
                            y10 += c*r10;
                            y20 += c*r20;
                        }
                        {
                            real & c = a1;
                            y11 += c * r10;
                            y20 += c * r21;
                        }
                        {
                            real & c = a4;
                            y10 += c * r11;
                            y21 += c * r20;
                        }
                        {
                            real & c = a2;
                            y12 += c * r10;
                            y20 += c * r22;
                        }
                        {
                            real& c = a8;
                            y10 += c * r12;
                            y22 += c* r20;
                        }
                        {
                            real & c = a3;
                            y13 += c * r10;
                            y20 += c * r23;
                        }
                        {
                            real & c  = aC;
                            y10 += c *r13;
                            y23 += c * r20;
                        }
                        {
                            real & c = a5;
                            y11+= c*r11;
                            y21 += c *r21;
                        }
                        {
                            real & c=  a6;
                            y12 += c*r11;
                            y21 += c*r22;
                        }
                        {
                            real & c = a9;
                            y11 += c *r12;
                            y22 += c *r21;
                        }
                        {
                            real &c  = a7;
                            y13 += c *r11;
                            y21 += c * r23;
                        }
                        {
                            real & c  = aD;
                            y11 +=c*r13;
                            y23 += c *r21;
                        }
                        {
                            real & c  = aA;
                            y12 += c * r12;
                            y22 += c* r22;
                        }
                        {
                            real & c = aE;
                            y12 += c * r13;
                            y23 += c * r22;
                        }
                        {
                            real & c = aB;
                            y13 += c * r12;
                            y22 += c *r23;
                        }
                        {
                            real & c = aF;
                            y13 += c * r13;
                            y23 += c *r23;
                        }
                        
                        Y1_[0] += y10;
                        Y1_[1] += y11;
                        Y1_[2] += y12;
                        Y1_[3] += y13;
                        Y2_[0] += y20;
                        Y2_[1] += y21;
                        Y2_[2] += y22;
                        Y2_[3] += y23;
                        
                        
                        
                    }
                    else
                    {
                        Matrix44* restored = new Matrix44(a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,aA,aB,aC,aD,aE,aF);
                        Vector4 res1=restored->vecmul(X1);
                        for(int i=0;i<4;i++)
                        {
                            Y1_[i] += res1[i];
                            
                        }
                    }
                }
                
                if(bloc_to_calculate.nbMatAdded >=3)
                {
                    int dist =bloc_to_calculate.index2_x - bloc_to_calculate.index2_y;
                    real a0 = bloc_to_calculate.a20;
                    real a1 = bloc_to_calculate.a21;
                    real a2 = bloc_to_calculate.a22;
                    real a3 = bloc_to_calculate.a23;
                    real a4 = bloc_to_calculate.a24;
                    real a5 = bloc_to_calculate.a25;
                    real a6 = bloc_to_calculate.a26;
                    real a7 = bloc_to_calculate.a27;
                    real a8 = bloc_to_calculate.a28;
                    real a9 = bloc_to_calculate.a29;
                    real aA = bloc_to_calculate.a2A;
                    real aB = bloc_to_calculate.a2B;
                    real aC = bloc_to_calculate.a2C;
                    real aD = bloc_to_calculate.a2D;
                    real aE = bloc_to_calculate.a2E;
                    real aF = bloc_to_calculate.a2F;
                    
                    
                    if(dist >=0)
                    {
                        X1 = X  + bloc_to_calculate.index2_y;
                        X2 = X  + bloc_to_calculate.index2_x;
                        Y1_= Y1 + bloc_to_calculate.index2_x;
                        Y2_= Y2 + bloc_to_calculate.index2_y;
                        
                    }
                    else
                    {
                        
                        X1 = X  + bloc_to_calculate.index2_x;
                        X2 = X  + bloc_to_calculate.index2_y;
                        Y2_= Y1 + bloc_to_calculate.index2_x; //Y2 "becomes" Y1
                        Y1_= Y2 + bloc_to_calculate.index2_y;
                        
                        
                    }
                    
                    if(dist !=0)
                    {
                        
                        real y10 = 0;
                        real y11 = 0;
                        real y12=0;
                        real y13=0;
                        real y20 = 0;
                        real y21 = 0;
                        real y22 = 0;
                        real y23=0;
                        real r10 = X1[0];
                        real r11 = X1[1];
                        real r12 = X1[2];
                        real r13 = X1[3];
                        real r20 = X2[0];
                        real r21 = X2[1];
                        real r22 = X2[2];
                        real r23 = X2[3];
                        {real & c = a0;
                            
                            y10 += c*r10;
                            y20 += c*r20;
                        }
                        {
                            real & c = a1;
                            y11 += c * r10;
                            y20 += c * r21;
                        }
                        {
                            real & c = a4;
                            y10 += c * r11;
                            y21 += c * r20;
                        }
                        {
                            real & c = a2;
                            y12 += c * r10;
                            y20 += c * r22;
                        }
                        {
                            real& c = a8;
                            y10 += c * r12;
                            y22 += c* r20;
                        }
                        {
                            real & c = a3;
                            y13 += c * r10;
                            y20 += c * r23;
                        }
                        {
                            real & c  = aC;
                            y10 += c *r13;
                            y23 += c * r20;
                        }
                        {
                            real & c = a5;
                            y11+= c*r11;
                            y21 += c *r21;
                        }
                        {
                            real & c=  a6;
                            y12 += c*r11;
                            y21 += c*r22;
                        }
                        {
                            real & c = a9;
                            y11 += c *r12;
                            y22 += c *r21;
                        }
                        {
                            real &c  = a7;
                            y13 += c *r11;
                            y21 += c * r23;
                        }
                        {
                            real & c  = aD;
                            y11 +=c*r13;
                            y23 += c *r21;
                        }
                        {
                            real & c  = aA;
                            y12 += c * r12;
                            y22 += c* r22;
                        }
                        {
                            real & c = aE;
                            y12 += c * r13;
                            y23 += c * r22;
                        }
                        {
                            real & c = aB;
                            y13 += c * r12;
                            y22 += c *r23;
                        }
                        {
                            real & c = aF;
                            y13 += c * r13;
                            y23 += c *r23;
                        }
                        
                        Y1_[0] += y10;
                        Y1_[1] += y11;
                        Y1_[2] += y12;
                        Y1_[3] += y13;
                        Y2_[0] += y20;
                        Y2_[1] += y21;
                        Y2_[2] += y22;
                        Y2_[3] += y23;
                        
                        
                        
                    }
                    else
                    {
                        Matrix44* restored = new Matrix44(a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,aA,aB,aC,aD,aE,aF);
                        Vector4 res1=restored->vecmul(X1);
                        for(int i=0;i<4;i++)
                        {
                            Y1_[i] += res1[i];
                            
                        }
                    }
                }
                
            }
            
            barrier2.wait();
            k++;
            
        }
        
    }
    void work6ntest(boost::barrier& barrier2, int thNb)
    {
        
        
        int k = 0;
        real* work;
        int work_nb;
        
        const real* X1;
        const real* X2;
        real* Y1_;
        real* Y2_;
        while(k<phase_number)
        {
            //std::cout<<"\nStart of phase"<<k<<"\n";
            
            _mutex.lock();
            work =  workingPhase6thway[k][thNb];
            work_nb = work_lengths[k][thNb];
            int* index = workingIndexes[k][thNb];
            _mutex.unlock();
            
            for(int m = 0; m<work_nb; m++)
            {
                
                
                int ix = index[m*2+0];
                int iy = index[m*2+1];
                
        
               
              
                real a0 = work[16*m+0];
                real a1 = work[16*m+1];
                real a2 = work[16*m+2];
                real a3 = work[16*m+3];
                real a4 = work[16*m+4];
                real a5 = work[16*m+5];
                real a6 = work[16*m+6];
                real a7 = work[16*m+7];
                real a8 = work[16*m+8];
                real a9 = work[16*m+9];
                real aA = work[16*m+10];
                real aB = work[16*m+11];
                real aC = work[16*m+12];
                real aD = work[16*m+13];
                real aE = work[16*m+14];
                real aF = work[16*m+15];
                
                if(ix-iy >=0)
                {
                    X1 = X  + iy;
                    X2 = X  + ix;
                    Y1_= Y1 + ix;
                    Y2_= Y2 + iy;
                    
                }
                else
                {
                    
                    X1 = X  + ix;
                    X2 = X  + iy;
                    Y2_= Y1 + ix;
                    Y1_= Y2 + iy;
                    
                    
                }
                
                if(ix-iy !=0)
                {
                    
                    real y10 = 0;
                    real y11 = 0;
                    real y12=0;
                    real y13=0;
                    real y20 = 0;
                    real y21 = 0;
                    real y22 = 0;
                    real y23=0;
                    real r10 = X1[0];
                    real r11 = X1[1];
                    real r12 = X1[2];
                    real r13 = X1[3];
                    real r20 = X2[0];
                    real r21 = X2[1];
                    real r22 = X2[2];
                    real r23 = X2[3];
                    {real & c = a0;
                        
                        y10 += c*r10;
                        y20 += c*r20;
                    }
                    {
                        real & c = a1;
                        y11 += c * r10;
                        y20 += c * r21;
                    }
                    {
                        real & c = a4;
                        y10 += c * r11;
                        y21 += c * r20;
                    }
                    {
                        real & c = a2;
                        y12 += c * r10;
                        y20 += c * r22;
                    }
                    {
                        real& c = a8;
                        y10 += c * r12;
                        y22 += c* r20;
                    }
                    {
                        real & c = a3;
                        y13 += c * r10;
                        y20 += c * r23;
                    }
                    {
                        real & c  = aC;
                        y10 += c *r13;
                        y23 += c * r20;
                    }
                    {
                        real & c = a5;
                        y11+= c*r11;
                        y21 += c *r21;
                    }
                    {
                        real & c=  a6;
                        y12 += c*r11;
                        y21 += c*r22;
                    }
                    {
                        real & c = a9;
                        y11 += c *r12;
                        y22 += c *r21;
                    }
                    {
                        real &c  = a7;
                        y13 += c *r11;
                        y21 += c * r23;
                    }
                    {
                        real & c  = aD;
                        y11 +=c*r13;
                        y23 += c *r21;
                    }
                    {
                        real & c  = aA;
                        y12 += c * r12;
                        y22 += c* r22;
                    }
                    {
                        real & c = aE;
                        y12 += c * r13;
                        y23 += c * r22;
                    }
                    {
                        real & c = aB;
                        y13 += c * r12;
                        y22 += c *r23;
                    }
                    {
                        real & c = aF;
                        y13 += c * r13;
                        y23 += c *r23;
                    }
                    
                    Y1_[0] += y10;
                    Y1_[1] += y11;
                    Y1_[2] += y12;
                    Y1_[3] += y13;
                    Y2_[0] += y20;
                    Y2_[1] += y21;
                    Y2_[2] += y22;
                    Y2_[3] += y23;
                    
                    
                   
                }
                else
                {
                    Matrix44* restored = new Matrix44(a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,aA,aB,aC,aD,aE,aF);
                    Vector4 res1=restored->vecmul(X1);
                    for(int i=0;i<4;i++)
                    {
                        Y1_[i] += res1[i];
                        
                    }
                }
            }
            
            barrier2.wait();
            k++;
        }
}
    void generateBlocks7thWay()
    {
        phase_tab = (int*) malloc(sizeof(int)*nbThreads);
        phase_tab[0] = 0;
        if(nbThreads %2 == 0)
        {
            phase_number = nbThreads /2 +1;
            for(int i =1; i<=nbThreads/2;i++)
            {
                phase_tab[i] = i;
                //std::cout << phase_tab[i] << " ";
            }
            for(int i=1 ; i<nbThreads/2;i++)
            {
                phase_tab[nbThreads/2 + i] = nbThreads/2 - i;
                //std::cout << phase_tab[nbThreads/2 + i] << " ";
            }
        }
        else
        {
            phase_number  = (nbThreads +1) / 2;
            for(int i =1; i<= (nbThreads - 1) /2; i++)
            {
                phase_tab[i] = 1;
                //std::cout << phase_tab[i] << " ";
            }
            for(int i=0; i< (nbThreads -1 )/ 2; i++)
            {
                phase_tab[(nbThreads-1)/2 +i] = (nbThreads-1)/2 -i;
                //std::cout << phase_tab[(nbThreads-1)/2  + i] << " ";
            }
        }
        
        workingPhase6thway = (real***) malloc(sizeof(real**)*phase_number);
        workingIndexes7 = (unsigned short***)malloc(sizeof(unsigned short**)*phase_number);
        firstBlocks = (int**)malloc(sizeof(int*)*phase_number);
        QueueN*** phasesTemp = (QueueN***) malloc(sizeof(QueueN**) * phase_number);
        QueueN*** phasesTempSwap = (QueueN***) malloc(sizeof(QueueN**) * phase_number);
        int threadBlockSize = (int) matrix->size() / (S_BLOCK_SIZE * nbThreads);
        int* phase_counter = (int*) malloc(sizeof(int)*phase_number);
        work_lengths = (int**) malloc(sizeof(int*)*phase_number);
        for(int i=0; i<phase_number;i++)
        {
            work_lengths[i] = (int*) malloc(sizeof(int)*nbThreads);
        }
        //Init
        
        for(int i =0; i<phase_number; i++)
        {
            workingPhase6thway[i] = (real**) malloc(sizeof(real*) * nbThreads);
            workingIndexes7[i] = (unsigned short**) malloc(sizeof(unsigned short*) * nbThreads);
            firstBlocks[i] = (int*)malloc(sizeof(int)*nbThreads);
            phasesTemp[i] = (QueueN**) malloc(sizeof(QueueN*)*nbThreads);
            phasesTempSwap[i] = (QueueN**) malloc(sizeof(QueueN*)*nbThreads);
            for(int j=0; j<nbThreads; j++)
            {
                phasesTemp[i][j] = new QueueN();
                phasesTempSwap[i][j] = new QueueN();
            }
        }
        
        int i =0;
        while(true)//Fetching blocks, assigned them to a phase each
        {
            int indX, indY;
            SparMatSymBlk::Column* col = &matrix->column_[i];
            if(col->nbb_>0)
            {
                //std::cout<<"Column number "<<matrix->colidx_[i]<<"->ind_YB : "<<(int) matrix->colidx_[i]/threadBlockSize<<"\n";
                indY = matrix->colidx_[i]/threadBlockSize;
                int indice_col = matrix->colidx_[i];
                
                for(int j=0; j<col->nbb_; j++)
                {
                    int indice_ligne = col->inx_[j];
                    //std::cout<<"Block position :"<<col->inx_[j]<<"-> ind_XB : "<<(int) col->inx_[j]/threadBlockSize<<"\n";
                    indX =col->inx_[j]/threadBlockSize;
                    //std::cout<<"Put in phase: "<<phase_tab[indX-indY]<<"\n";
                    int swap = 0;
                    if(indX - indY >= 1+ (int)(nbThreads/2))
                    {
                        swap =1;
                    }
                    //swap = 0; // Le swap ne fonctionne pas encore.
                    int phase =phase_tab[indX-indY];
                    if(phase>0)
                    {
                        if(swap ==1)
                        {
                            // std::cout<<"\nWARNINGINININING\n";
                        }
                    }
                    //phasesTemp[phase][phase_counter[phase]%nbThreads]->push_front(Tuple2( indice_ligne, indice_col,swap, &matrix->block(col->inx_[j],matrix->colidx_[i])));
                    mytuplenew data;
                    Matrix44 matrix= col->blk_[j];
                    data.a0 = matrix.val[0x0];
                    data.a1 = matrix.val[0x1];
                    data.a2 = matrix.val[0x2];
                    data.a3 = matrix.val[0x3];
                    data.a4 = matrix.val[0x4];
                    data.a5 = matrix.val[0x5];
                    data.a6 = matrix.val[0x6];
                    data.a7 = matrix.val[0x7];
                    data.a8 = matrix.val[0x8];
                    data.a9 = matrix.val[0x9];
                    data.aA = matrix.val[0xA];
                    data.aB = matrix.val[0xB];
                    data.aC = matrix.val[0xC];
                    data.aD = matrix.val[0xD];
                    data.aE = matrix.val[0xE];
                    data.aF = matrix.val[0xF];
                    
                    data.index_x = indice_ligne * blocksize;
                    data.index_y = indice_col * blocksize;
                    
                    if(swap==0)
                    {
                        //std::cout<<data.index_x<<","<<data.index_y<<"\n";
                        phasesTemp[phase][indX]->push_front(data);
                    }
                    else
                    {
                        //std::cout<<data.index_x<<","<<data.index_y<<"\n";
                        int temp = data.index_x;
                        data.index_x = data.index_y;
                        data.index_y = temp;
                        
                        phasesTemp[phase][indY]->push_front(data);
                    }
                    
                    phase_counter[phase]++;
                    
                }
                //std::cout<<"\n\n";
            }
            
            i = matrix->colidx_[i+1];
            if(i>=matrix->rsize_)
            {
                
                break;
            }
            
        }
        
        int relevantSize = (int) matrix->size() / nbThreads;
        for(int i=0; i<phase_number; i++)//Putting in order for threads
        {
            std::cout<<"New phase";
            for(int j=0; j<nbThreads;j++)
            {
                
                
                int t_work_length = (int)phasesTemp[i][j]->size();
                
                
                work_lengths[i][j] = t_work_length;
                
                //std::cout<<"wl1:"<<t_work_length1<<" wl2:"<<t_work_length2<<"\n";
                workingPhase6thway[i][j] = (real*) malloc(sizeof(real)*t_work_length*16);
                workingIndexes7[i][j] = (unsigned short*) malloc(sizeof(unsigned short)*t_work_length*2);
                
                //threadBlockSize
                
                //xi + yi*(thread_blocksize * blocksize) = ni
                //xi = ni%(thead_blocksize * blocksize)
                //yi = ni/(thread_blocksize * blocksize)
                //xi+1 - xi + (yi+1 - yi)*thread_blocksize = ni+1 - ni
                //
                
                int last_x = 0;
                int last_y =0;
                for(int m=0; m<t_work_length; m++)//ajouter les blocks swappes
                {
                    
                   
                    mytuplenew first= phasesTemp[i][j]->back();
                    phasesTemp[i][j]->pop_back();
                    
                    workingPhase6thway[i][j][16*m] = first.a0;
                    workingPhase6thway[i][j][16*m+1] = first.a1;
                    workingPhase6thway[i][j][16*m+2] = first.a2;
                    workingPhase6thway[i][j][16*m+3] = first.a3;
                    workingPhase6thway[i][j][16*m+4] = first.a4;
                    workingPhase6thway[i][j][16*m+5] = first.a5;
                    workingPhase6thway[i][j][16*m+6] = first.a6;
                    workingPhase6thway[i][j][16*m+7] = first.a7;
                    workingPhase6thway[i][j][16*m+8] = first.a8;
                    workingPhase6thway[i][j][16*m+9] = first.a9;
                    workingPhase6thway[i][j][16*m+10] = first.aA;
                    workingPhase6thway[i][j][16*m+11] = first.aB;
                    workingPhase6thway[i][j][16*m+12] = first.aC;
                    workingPhase6thway[i][j][16*m+13] = first.aD;
                    workingPhase6thway[i][j][16*m+14] = first.aE;
                    workingPhase6thway[i][j][16*m+15] = first.aF;
                    
                    if(m !=0)
                    {
                        if(first.index_y <= first.index_x)
                        {
                            workingIndexes7[i][j][m] = first.index_x-last_x + (first.index_y-last_y) * relevantSize;
                        }
                        else
                        {
                            workingIndexes7[i][j][m] = first.index_y-last_y + (first.index_x-last_x) * relevantSize;
                        }
                    }
                else
                    {
                        workingIndexes7[i][j][m] = 0;
                        firstBlocks[i][j] = first.index_x + 65535*first.index_y;
                    }
                if(j>100)
                    {
                        
                            // std::cout<<first.index_x-last_x + (first.index_y-last_y) * relevantSize<<"\n";
                            std::cout<<"Block sent: ("<<first.index_x<<","<<first.index_y<<")"<<"pad:"<< workingIndexes7[i][j][m]<<"th:"<<j<<"\n";
                            
                            if(m ==0)
                            {
                                std::cout<<"value sent :"<<first.index_x + 65535*first.index_y<<"\n";
                            }
                        
                    }
                  //  std::cout<<"lastix:"<<last_x<<"lastiy:"<<last_y<<"\n";
                    
                   // std::cout<<"ix:"<<first.index_x<<"iy:"<<first.index_y<<"\n\n";
                    last_x = first.index_x;
                    last_y = first.index_y;
                    
                }
                
               
                
                
            }
        }
    }
    
    
    
    void work7ntest(boost::barrier& barrier2, int thNb)
    {
        
        
        int k = 0;
        real* work;
        int work_nb;
        int relevantSize = (int) matrix->size() / nbThreads;
        const real* X1;
        const real* X2;
        real* Y1_;
        real* Y2_;
        real y20 = 0;
        real y21 = 0;
        real y22 = 0;
        real y23=0;
        real y10 = 0;
        real y12=0;
        real y13=0;
        real y11 = 0;
        real r10 = 0;
        real r11 = 0;
        real r12 = 0;
        real r13 = 0;
        real r20 = 0;
        real r21 = 0;
        real r22 = 0;
        real r23 = 0;
        while(k<phase_number)
        {
            //std::cout<<"\nStart of phase"<<k<<"\n";
            
            _mutex.lock();
            work =  workingPhase6thway[k][thNb];
            work_nb = work_lengths[k][thNb];
            unsigned short* index = workingIndexes7[k][thNb];
            
            int ix_f = firstBlocks[k][thNb]%65535;
            int iy_f = firstBlocks[k][thNb]/65535;
            _mutex.unlock();
            
            int pad = 0;
            short dist = ix_f - iy_f;
            bool change;
            int ix = 0;
            int iy =0;
           
            for(int m = 0; m<work_nb; m++)
            {
                
            
                    pad += index[m];
                    
                
                    if(dist <0)
                    {
                        iy = iy_f+ pad%relevantSize;
                        change = (iy+index[m]-iy_f >= relevantSize) || m ==0;
                        ix = ix_f+ (int) pad/relevantSize;
                        
                    }
                    else
                    {
                        change = (ix+index[m]-ix_f >= relevantSize) || m ==0;
                        ix = ix_f+ pad%relevantSize;
                        iy = iy_f+ (int) pad/relevantSize;
                    }
               
                
        
               
                
                real a0 = work[16*m+0];
                real a1 = work[16*m+1];
                real a2 = work[16*m+2];
                real a3 = work[16*m+3];
                real a4 = work[16*m+4];
                real a5 = work[16*m+5];
                real a6 = work[16*m+6];
                real a7 = work[16*m+7];
                real a8 = work[16*m+8];
                real a9 = work[16*m+9];
                real aA = work[16*m+10];
                real aB = work[16*m+11];
                real aC = work[16*m+12];
                real aD = work[16*m+13];
                real aE = work[16*m+14];
                real aF = work[16*m+15];
               
                if(m!=0)
                    {
                        if(dist>=0)//Cas de base, on itère sur x
                        {
                            if(!change)
                                {
                                    
                                }
                                else//Je change Y2 et X1, j'applique donc les résultats des calculs précendents
                                {
                                    Y2_[0] += y20;
                                    Y2_[1] += y21;
                                    Y2_[2] += y22;
                                    Y2_[3] += y23;
                                    Y2_= Y2 + iy;
                                    X1 = X  + iy;
                                    y20 = 0;
                                    y21 = 0;
                                    y22 = 0;
                                    y23=0;
                                    r10 = X1[0];
                                    r11 = X1[1];
                                    r12 = X1[2];
                                    r13 = X1[3];
                                }
                                Y1_[0] += y10;
                                Y1_[1] += y11;
                                Y1_[2] += y12;
                                Y1_[3] += y13;
                                X2 = X  + ix;
                                Y1_= Y1 + ix;
                                r20 = X2[0];
                                r21 = X2[1];
                                r22 = X2[2];
                                r23 = X2[3];
                                y10 =0;
                                y11 =0;
                                y12=0;
                                y13=0;
                            
                            
                            }
                        else
                        {
                            
                            
                                if(!change)//Je change Y2 et X1, j'applique donc les résultats des calculs précendents
                                {}
                                else
                                {
                                    
        
                                    Y2_[0] += y20;
                                    Y2_[1] += y21;
                                    Y2_[2] += y22;
                                    Y2_[3] += y23;
                                    
                                    X1 = X  + ix;
                                    Y2_= Y1 + ix;
                                    r10 = X1[0];
                                    r11 = X1[1];
                                    r12 = X1[2];
                                    r13 = X1[3];
                                    y20 = 0;
                                    y21 = 0;
                                    y22 = 0;
                                    y23=0;
                                    
                                }
                                Y1_[0] += y10;
                                Y1_[1] += y11;
                                Y1_[2] += y12;
                                Y1_[3] += y13;
                                X2 = X  + iy;
                                Y1_= Y2 + iy;
                                r20 = X2[0];
                                r21 = X2[1];
                                r22 = X2[2];
                                r23 = X2[3];
                                y10 = 0;
                                y12=0;
                                y13=0;
                                y11 = 0;
                                
                            
                        
                            
                            
                            
                            
                        }
                    }
                else
                    {
                        if(dist>=0)//Cas de base, on itère sur x
                        {
                            
                            Y2_= Y2 + iy;
                            X1 = X  + iy;
                            X2 = X  + ix;
                            Y1_= Y1 + ix;
                            y10 =0;
                            y11 =0;
                            y12=0;
                            y13=0;
                            y20 = 0;
                            y21 = 0;
                            y22 = 0;
                            y23=0;
                            r10 = X1[0];
                            r11 = X1[1];
                            r12 = X1[2];
                            r13 = X1[3];
                            r20 = X2[0];
                            r21 = X2[1];
                            r22 = X2[2];
                            r23 = X2[3];
                        }
                        else
                        {
                            X2 = X  + iy;
                            Y1_= Y2 + iy;
                            X1 = X  + ix;
                            Y2_= Y1 + ix;
                            y10 = 0;
                            y12=0;
                            y13=0;
                            y11 = 0;
                            y20 = 0;
                            y21 = 0;
                            y22 = 0;
                            y23=0;
                            r10 = X1[0];
                            r11 = X1[1];
                            r12 = X1[2];
                            r13 = X1[3];
                            r20 = X2[0];
                            r21 = X2[1];
                            r22 = X2[2];
                            r23 = X2[3];
                        }
                    }
                if(ix-iy !=0)
                {
                    
    
                    {real & c = a0;
                        
                        y10 += c*r10;
                        y20 += c*r20;
                    }
                    {
                        real & c = a1;
                        y11 += c * r10;
                        y20 += c * r21;
                    }
                    {
                        real & c = a4;
                        y10 += c * r11;
                        y21 += c * r20;
                    }
                    {
                        real & c = a2;
                        y12 += c * r10;
                        y20 += c * r22;
                    }
                    {
                        real& c = a8;
                        y10 += c * r12;
                        y22 += c* r20;
                    }
                    {
                        real & c = a3;
                        y13 += c * r10;
                        y20 += c * r23;
                    }
                    {
                        real & c  = aC;
                        y10 += c *r13;
                        y23 += c * r20;
                    }
                    {
                        real & c = a5;
                        y11+= c*r11;
                        y21 += c *r21;
                    }
                    {
                        real & c=  a6;
                        y12 += c*r11;
                        y21 += c*r22;
                    }
                    {
                        real & c = a9;
                        y11 += c *r12;
                        y22 += c *r21;
                    }
                    {
                        real &c  = a7;
                        y13 += c *r11;
                        y21 += c * r23;
                    }
                    {
                        real & c  = aD;
                        y11 +=c*r13;
                        y23 += c *r21;
                    }
                    {
                        real & c  = aA;
                        y12 += c * r12;
                        y22 += c* r22;
                    }
                    {
                        real & c = aE;
                        y12 += c * r13;
                        y23 += c * r22;
                    }
                    {
                        real & c = aB;
                        y13 += c * r12;
                        y22 += c *r23;
                    }
                    {
                        real & c = aF;
                        y13 += c * r13;
                        y23 += c *r23;
                    }
                    
                }
                else
                {
                    Matrix44* restored = new Matrix44(a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,aA,aB,aC,aD,aE,aF);
                    Vector4 res1=restored->vecmul(X1);
                    for(int i=0;i<4;i++)
                    {
                        Y1_[i] += res1[i];
                        
                    }
                }
                if(m!= work_nb-1)
                {}
                else
                {
                    Y1_[0] += y10;
                    Y1_[1] += y11;
                    Y1_[2] += y12;
                    Y1_[3] += y13;
                    Y2_[0] += y20;
                    Y2_[1] += y21;
                    Y2_[2] += y22;
                    Y2_[3] += y23;
                }
            }
            
            barrier2.wait();
            k++;
        }
}
        
    
    
    void vecMulAddnTimes(const real*X_calc, real*Y, int n_time)
    {
        //Start thread,
        //Calculate, wait, calculate, wait..
        X= X_calc;
        //generateBlocks3rdWay();
        generateBlocks4thWay();
        try
        {
            boost::thread* threads = (boost::thread*) malloc(sizeof(boost::thread)*nbThreads);
            boost::barrier bar(nbThreads);
            boost::barrier bar2(nbThreads);
            for(int i=0;i<nbThreads; i++)
            {
                threads[i] = boost::thread(boost::bind(&MatSymBMtInstance::work2m, this,boost::ref(bar),boost::ref(bar2), n_time));;
            }
            
            std::cout<<"Thread started\n";
            
            
           
            
            for(int i=0;i<nbThreads; i++)
            {
                threads[i].join();
            }
            for(int i=0; i<nbThreads;i++)
            {
                threads[i].interrupt();
            }
        }
        catch (const std::exception &e)
        {
            std::cerr << "Exception occurred: " << e.what() << std::endl;
            // Handle exception here if needed
            
        }
        for(int i=0;i<big_mat_size;i++)
        {
                
                Y[i]+= Y1[i]+Y2[i];
                
        }
        
    
    }
    void vecMulAddnTimes2(const real*X_calc, real*Y, int n_time)
    {
        //Start thread,
        //Calculate, wait, calculate, wait..
        X= X_calc;
        generateBlocks3rdWay();
        //generateBlocks4thWay();
        try
        {
            boost::thread* threads = (boost::thread*) malloc(sizeof(boost::thread)*nbThreads);
            boost::barrier bar(nbThreads);
            boost::barrier bar2(nbThreads);
            for(int i=0;i<nbThreads; i++)
            {
                threads[i] = boost::thread(boost::bind(&MatSymBMtInstance::work3m, this,boost::ref(bar),boost::ref(bar2), n_time));;
            }
            
            std::cout<<"Thread started\n";
            
            
           
            
            for(int i=0;i<nbThreads; i++)
            {
                threads[i].join();
            }
            for(int i=0; i<nbThreads;i++)
            {
                threads[i].interrupt();
            }
        }
        catch (const std::exception &e)
        {
            std::cerr << "Exception occurred: " << e.what() << std::endl;
            // Handle exception here if needed
            
        }
        for(int i=0;i<big_mat_size;i++)
        {
                
                Y[i]+= Y1[i]+Y2[i];
                
        }
        
    
    }
    void vecMulAddnTimes3(const real*X_calc, real*Y, int n_time)
    {
        //Start thread,
        //Calculate, wait, calculate, wait..
        X= X_calc;
        generateBlocks5thWay();
        //generateBlocks4thWay();
        try
        {
            boost::thread* threads = (boost::thread*) malloc(sizeof(boost::thread)*nbThreads);
            boost::barrier bar(nbThreads);
            boost::barrier bar2(nbThreads);
            for(int i=0;i<nbThreads; i++)
            {
                threads[i] = boost::thread(boost::bind(&MatSymBMtInstance::work4m, this,boost::ref(bar),boost::ref(bar2), n_time));;
            }
            
            std::cout<<"Thread started\n";
            
            
           
            
            for(int i=0;i<nbThreads; i++)
            {
                threads[i].join();
            }
            for(int i=0; i<nbThreads;i++)
            {
                threads[i].interrupt();
            }
        }
        catch (const std::exception &e)
        {
            std::cerr << "Exception occurred: " << e.what() << std::endl;
            // Handle exception here if needed
            
        }
        for(int i=0;i<big_mat_size;i++)
        {
                
                Y[i]+= Y1[i]+Y2[i];
                
        }
        
    
    }
    void vecMulAddnTimes5(const real*X_calc, real*Y, int n_time)
    {
        //Start thread,
        //Calculate, wait, calculate, wait..
        X= X_calc;
        generateBlocks7thWay();
        //generateBlocks4thWay();
        try
        {
            boost::thread* threads = (boost::thread*) malloc(sizeof(boost::thread)*nbThreads);
            boost::barrier bar(nbThreads);
            boost::barrier bar2(nbThreads);
            for(int i=0;i<nbThreads; i++)
            {
                threads[i] = boost::thread(boost::bind(&MatSymBMtInstance::work7m, this,boost::ref(bar),boost::ref(bar2), n_time));;
            }
            
            std::cout<<"Thread started\n";
            
            
           
            
            for(int i=0;i<nbThreads; i++)
            {
                threads[i].join();
            }
            for(int i=0; i<nbThreads;i++)
            {
                threads[i].interrupt();
            }
        }
        catch (const std::exception &e)
        {
            std::cerr << "Exception occurred: " << e.what() << std::endl;
            // Handle exception here if needed
            
        }
        for(int i=0;i<big_mat_size;i++)
        {
                
                Y[i]+= Y1[i]+Y2[i];
                
        }
        
    
    }
    void vecMulAddnTimes4(const real*X_calc, real*Y, int n_time)
    {
        //Start thread,
        //Calculate, wait, calculate, wait..
        X= X_calc;
        generateBlocks6thWay();
        //generateBlocks4thWay();
        try
        {
            boost::thread* threads = (boost::thread*) malloc(sizeof(boost::thread)*nbThreads);
            boost::barrier bar(nbThreads);
            boost::barrier bar2(nbThreads);
            for(int i=0;i<nbThreads; i++)
            {
                threads[i] = boost::thread(boost::bind(&MatSymBMtInstance::work6m, this,boost::ref(bar),boost::ref(bar2), n_time));;
            }
            
            std::cout<<"Thread started\n";
            
            
           
            
            for(int i=0;i<nbThreads; i++)
            {
                threads[i].join();
            }
            for(int i=0; i<nbThreads;i++)
            {
                threads[i].interrupt();
            }
        }
        catch (const std::exception &e)
        {
            std::cerr << "Exception occurred: " << e.what() << std::endl;
            // Handle exception here if needed
            
        }
        for(int i=0;i<big_mat_size;i++)
        {
                
                Y[i]+= Y1[i]+Y2[i];
                
        }
        
    
    }
};

#endif /* MatSymBMtInstance_hpp */


