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
IndependentTesting <- "A549"
IndependentTesting.index <- which(colnames(Base_relsotionSite.csv)==IndependentTesting)
IndependentTesting.row <- which(Base_relsotionSite.csv[,IndependentTesting.index]==1)
IndependentTesting.data <- Base_relsotionSite.csv[IndependentTesting.row,c(2:6)]

##Find training data record
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
Training.maturemRNA.geneInformation <- subsetByOverlaps(genes(txdb),Training.GR.ConsensusM6A)
IndependentTesting.GR.ConsensusM6A.geneInformation <- subsetByOverlaps(genes(txdb),IndependentTesting.GR.ConsensusM6A)

#Find the motif in training dataset or independent dataset
MotifInmaturemRNA <- m6ALogisticModel::sample_sequence("RRACH",Training.FullTranscript.geneInformation,Hsapiens). # if generate training dataset, choose this
#MotifInmaturemRNA <- m6ALogisticModel::sample_sequence("RRACH",IndependentTesting.GR.ConsensusM6A.geneInformation,Hsapiens). # if generating independent dataset, choose this

#Get the gene for negative samples from exons regions.
NotMethylated.gene <- subsetByOverlaps(MotifInmaturemRNA,exons(txdb))

#remove the motif in positive samples.
NotMethylated.motif <- NotMethylated.gene[!NotMethylated.gene %over% c(Training.GR.ConsensusM6A)]
#NotMethylated.motif <- NotMethylated.gene[!NotMethylated.gene %over% c(IndependentTesting.GR.ConsensusM6A)]

Neg.GR <- NotMethylated.motif-2
Neg.predictor.Seqdata <- as.character(DNAStringSet(Views(Hsapiens,Neg.GR+500)))

#Sample from gene
Training.GR <- subsetByOverlaps(IndependentTesting.GR.ConsensusM6A,exons(txdb)) # remove the data in independent dataset
Training.predictor.Seqdata <- as.character(DNAStringSet(Views(Hsapiens,Training.GR+500)))

#save the data for independent dataset.
write.table(Training.predictor.Seqdata,"maturemRNA/Pos_MOLM13_test_mature_seq.csv",row.names=FALSE,col.names=FALSE,sep=",")
write.table(Neg.predictor.Seqdata,"maturemRNA/Pos_MOLM13_test_mature_seq.csv",row.names=FALSE,col.names=FALSE,sep=",")

# write.table(Training.predictor.Seqdata,"maturemRNA/Pos_MOLM13_train_mature_seq.csv",row.names=FALSE,col.names=FALSE,sep=",")
# write.table(Neg.predictor.Seqdata,"maturemRNA/Pos_MOLM13_train_mature_seq.csv",row.names=FALSE,col.names=FALSE,sep=",")