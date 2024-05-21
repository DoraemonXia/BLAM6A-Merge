#import packages.
require(GenomicAlignments)
require(Rsamtools)
require(BiocParallel)
require(BSgenome)
require(BSgenome.Hsapiens.UCSC.hg19)
library(TxDb.Hsapiens.UCSC.hg19.knownGene)
library(BSgenome)
library(dplyr)
library(matrixStats)
library(m6ALogisticModel)
library(SummarizedExperiment)
library(TxDb.Hsapiens.UCSC.hg19.knownGene)
library(BSgenome.Hsapiens.UCSC.hg19)
library(fitCons.UCSC.hg19)
library(phastCons100way.UCSC.hg19)
library(dplyr)
library(caret)
library(e1071)
library(graphics)
library(pROC)
library(ROCR)

#read_source_file
Base_relsotionSite.csv <- read.csv("SingleNucleotideReselutionData.csv")
GR <- function(Data){
  data <- GRanges(seqnames = Data$chromsome,
                  ranges = IRanges(start = Data$modEnd,width = 1),
                  strand = Data$strand)
  return(data)
}

#choose which cell data to be independent dataset
IndependentTesting <- "MOLM13"
IndependentTesting.index <- which(colnames(Base_relsotionSite.csv)==IndependentTesting)
IndependentTesting.row <- which(Base_relsotionSite.csv[,IndependentTesting.index]==1)
IndependentTesting.data <- Base_relsotionSite.csv[IndependentTesting.row,c(2:6)]

#Find training data record
Training.data.index.col <- c(1:12)[-IndependentTesting.index][-c(1:6)]
Training.data <- Base_relsotionSite.csv[,c(Training.data.index.col)]
Training.data.index.row <- apply(Training.data, 1, sum)
Training.MoreThanTwoRecorded <- which(Training.data.index.row>1)
Training.GR <- GR(Base_relsotionSite.csv[Training.MoreThanTwoRecorded,c(2:6)])
IndependentTesting.GR <- GR(IndependentTesting.data)
Hsapiens <- BSgenome.Hsapiens.UCSC.hg19

#put in gene data, and find the "RRACH" motif in all relevant genes
Training.GR.ConsensusM6A <- Training.GR[which(vcountPattern("RRACH",DNAStringSet(Views(Hsapiens, Training.GR+2)),fixed = F)==1)]
IndependentTesting.GR.ConsensusM6A <- IndependentTesting.GR[which(vcountPattern("RRACH",DNAStringSet(Views(Hsapiens, IndependentTesting.GR+2)),fixed = F)==1)]
txdb <- TxDb.Hsapiens.UCSC.hg19.knownGene
Training.FullTranscript.geneInformation <- subsetByOverlaps(genes(txdb),Training.GR.ConsensusM6A)
IndependentTesting.GR.ConsensusM6A.geneInformation <- subsetByOverlaps(genes(txdb),IndependentTesting.GR.ConsensusM6A)

#Find the motif in training dataset
#MotifInFullTranscript <- m6ALogisticModel::sample_sequence("RRACH",Training.FullTranscript.geneInformation,Hsapiens)

#Find the motif in independent dataset
MotifInFullTranscript <- m6ALogisticModel::sample_sequence("RRACH",IndependentTesting.GR.ConsensusM6A.geneInformation,Hsapiens)

#Get the gene for negative samples
NotMethylated.gene <- subsetByOverlaps(MotifInFullTranscript,genes(txdb))

#remove the motif in positive samples.
#NotMethylated.motif <- NotMethylated.gene[!NotMethylated.gene %over% c(Training.GR.ConsensusM6A)]
NotMethylated.motif <- NotMethylated.gene[!NotMethylated.gene %over% c(IndependentTesting.GR.ConsensusM6A)]

#remove the pattern of 'NNNNN'
NotMethylated.motif <- NotMethylated.motif[-which(vcountPattern("NNNNN",DNAStringSet(Views(Hsapiens, NotMethylated.motif)),fixed = T)==1)]

#Sample from gene
NotMethylated.motif <- sample(NotMethylated.motif,400000)
Neg.GR <- NotMethylated.motif-2
Neg.predictor.Seqdata <- as.character(DNAStringSet(Views(Hsapiens,Neg.GR+500)))

#request the sequences in gene for training or independent dataset.
#Training.GR <- subsetByOverlaps(Training.GR.ConsensusM6A,genes(txdb))
Training.GR <- subsetByOverlaps(IndependentTesting.GR.ConsensusM6A,genes(txdb))
Training.predictor.Seqdata <- as.character(DNAStringSet(Views(Hsapiens,Training.GR+500)))

#save the data for independent dataset.
write.table(Training.predictor.Seqdata,"FullTrans/Pos_MOLM13_test_FullTranscript_seq.csv",row.names=FALSE,col.names=FALSE,sep=",")
write.table(Neg.predictor.Seqdata,"FullTrans/Neg_MOLM13_test_FullTranscript_seq.csv",row.names=FALSE,col.names=FALSE,sep=",")

# write.table(Training.predictor.Seqdata,"FullTrans/Pos_MOLM13_train_FullTranscript_seq.csv",row.names=FALSE,col.names=FALSE,sep=",")
# write.table(Neg.predictor.Seqdata,"FullTrans/Neg_MOLM13_train_FullTranscript_seq.csv",row.names=FALSE,col.names=FALSE,sep=",")