library(ggplot2)
library(jsonlite)
library(reshape2)
library(scales)
library(ggpubr)

################################################################################

targetMetricsName <- "accuracy"
subGroupNames <- c("Train (words)", 
                   "Test (words)", 
                   "Train (counts)", 
                   "Test (counts)")

statsFileReport <- list("bow" = c("Train (words)", "Test (words)"),
                        "counts" = c("Train (counts)", "Test (counts)"))

## ORIGINAL
setwd("path/to/results/")
dataResultsFileNames <- c("results_linear_MOST_FREQUENT.json", 
                          "results_linear_MOST_FREQUENT_FROM_EACH_TEXT.json", 
                          "results_linear_RANDOM.json")

## FIXED TRAIN SIZE
setwd("path/to/results/fixed_ratio/")
dataResultsFileNames <- c("results_fixed_train_MOST_FREQUENT.json", 
                          "results_fixed_train_MOST_FREQUENT_FROM_EACH_TEXT.json", 
                          "results_fixed_train_RANDOM.json")

################################################################################

toNiceTitle <- function(pairName, observationsCount){
    title <- gsub("_", " ", gsub("vs_", "/\n", pairName))
    title <- gsub("plos", "articles", title)
}

removeXAxis <- function(p, removeLabs, newYLab){
    if (FALSE){
    #if (removeLabs == TRUE){
        p <- p + theme(
            axis.text.x = element_blank(),
            axis.ticks.x = element_blank(),
            axis.title.x = element_blank(),
        )
    }
    
    p <- p + scale_y_continuous(limits=c(0.40, 1),oob = rescale_none, breaks=c(0.50, 0.75, 1))
    p <- p + ylab(paste0(newYLab, " [tokens]"))
    p <- p + theme( plot.margin = margin(0, 0, 0, 0) )
    p
}

for (dataFileName in dataResultsFileNames){
    #dataFileName <- dataResultsFileNames[[3]]
    data <- fromJSON(dataFileName)
    pairNames <- names(data)
    existingTargetLens <- names(data[[1]])
    statsAll <- list()
    plotCompound <- list()
    
    for(targetLen in existingTargetLens){
        meltedResultsDf <- c()
        dfObservationsCount <- data.frame()
        numberOfResamples <- length( data[[1]][[targetLen]][["train"]][[1]] )
        statsAll[[targetLen]] <- list()
        
        for(pairName in pairNames){
            df <- data.frame( matrix(NA, numberOfResamples, length(subGroupNames)) )
            colnames(df) <- subGroupNames
            
            observationsCount <- data[[pairName]][[targetLen]][["lesser_num_of_texts"]]
            nicePairName      <- toNiceTitle(pairName, observationsCount)
            df$pairName       <- nicePairName
            pairHasResults    <- length(data[[pairName]][[targetLen]][["train"]][[1]]) > 0
            
            if ( pairHasResults == TRUE ){
                df[,"Train (words)"]    <- data[[pairName]][[targetLen]][["train"]][[targetMetricsName]]
                df[,"Test (words)"]     <- data[[pairName]][[targetLen]][["test"]][[targetMetricsName]]
                df[,"Train (counts)"]   <- data[[pairName]][[targetLen]][["train"]][["accuracy_counts"]]
                df[,"Test (counts)"]    <- data[[pairName]][[targetLen]][["test"]][["accuracy_counts"]]
            }
            
            meltedDf <- melt(df, id="pairName")
            meltedResultsDf <- rbind(meltedResultsDf, meltedDf)
            dfObservationsCount[nicePairName, "n"] <- observationsCount
            statsAll[[targetLen]][[pairName]] <- df
        }
        
        ## PLOT ################################################################
        size <- 0.5
        dfObservationsCount$pairName <- rownames(dfObservationsCount)
        
        image <- ggplot(meltedResultsDf, 
                        aes(x=pairName, 
                            y=value, 
                            fill=variable, 
                            colour=variable)) +
            stat_summary(fun=mean, geom="col", position=position_dodge(size), width=size, alpha=0.100) +
            stat_summary(fun.data=mean_cl_normal,
                         geom="errorbar", 
                         size=1.2, 
                         position=position_dodge(size),
                         width=size) +
            geom_hline(aes(yintercept=0.50), linetype="longdash") +
            geom_text(aes(0,0.50,label = "Random model Accuracy", vjust = 1.618, hjust=-0.1), color="grey54") +
            theme_bw() +
            theme(#text = element_text(size = 13), 
                  legend.position = "bottom",
                  axis.text.x = element_text(angle = 0),
                  axis.title.x=element_blank(),
                  #panel.grid.major.x = element_blank(),
                  axis.title.y  = element_text(size=11)
            ) +
            xlab("Pair") +
            ylab("Accuracy") +
            labs(color="Dataset") + 
            labs(fill="Dataset")+
            geom_label(inherit.aes = FALSE, 
                       data=dfObservationsCount, 
                       aes(label=paste0(n, " Obs."), 
                           x=pairName), 
                       y=0,
                       size=3.333,
                       color="gray20")
        
        image
        
        dir.create( paste0("results/", dataFileName) )
        fileName <- paste0("results/", dataFileName, "/", dataFileName, "_targetLen_", targetLen, ".png" )
        #ggsave(file=fileName, plot=image, width=11.5, height=6)
        ggsave(file=fileName, plot=image, width=9, height=4)
        plotCompound[[targetLen]] <- image
    }
    
    library(ggpubr)
    for(i in 1:(length(plotCompound) )){
        lengthTitle <- names(plotCompound)[i]
        plotCompound[[i]] <- removeXAxis(plotCompound[[i]], i != length(plotCompound), lengthTitle)
    }
    
    plotCompound[["nrow"]] <- length(plotCompound)
    plotCompound[["ncol"]] <- 1
    plotCompound[["common.legend"]] <- TRUE
    plotCompound[["legend"]] <- "bottom"
    plotCompound[["align"]] <- "hv"
    plotCompound[["heights"]] <- rep(0.05, 5)
    
    p <- do.call("ggarrange", plotCompound)
    p <- annotate_figure(p, left=
                             text_grob("Accuracy",  size = 14, family = "Arial", rot=90)
                         )

    fileName <- paste0("results/", dataFileName, "/", dataFileName, "_all.png" )
    ggsave(file=fileName, plot=p, width=9, height= 2.61 * plotCompound[["nrow"]])
    
    
    ## STATS TABLES ############################################################
    
    statsResults <- list()
    statsResultsRawTrain <- list()
    statsResultsRawTest  <- list()

    for(reportStatsCategory in names(statsFileReport)){
        reportColumns <- statsFileReport[[reportStatsCategory]]

        for(targetLen in names(statsAll)){
            for(pairName in names(statsAll[[targetLen]])){
                partialResults <- apply( statsAll[[targetLen]][[pairName]][,reportColumns], 2, function(column){ 
                    ROUND_DIGITS <- 1
                    strStats <- "/NA/"
                    column <- na.omit(column)
                    estimate <- NA
                    
                    if (length(column) > 0){
                        if ( sd(column) > 0 ){
                            stats <- t.test( column )
                            estimate <- round( stats$estimate * 100, ROUND_DIGITS )
                            CI <- round( as.vector( stats$conf.int * 100), ROUND_DIGITS)
                            
                            strCI <- paste0 (CI, collapse = ", " )
                            strStats <- paste0( c(estimate, " (", strCI, ")"), collapse = "")
                            #strStats <- gsub("\\b0\\.", ".", strStats)
                        }else{
                            estimate <- mean( column ) * 100
                            strStats <- round( estimate, ROUND_DIGITS )
                        }
                    }
                    
                    return( list("str" = strStats, "raw" = estimate ) )
                })
                
                partialResults <- as.vector(partialResults)
                trainResult <- partialResults[[1]][["str"]]
                testResults <- partialResults[[2]][["str"]]
                
                keyTrain <- paste0(pairName, "_train")
                keyTest  <- paste0(pairName, "_test")
                statsResults[[keyTrain]][[targetLen]] <- paste0(trainResult, collapse="#")
                statsResults[[keyTest]][[targetLen]]  <- paste0(testResults, collapse="#")
                
                statsResultsRawTrain[[pairName]][[targetLen]] <- partialResults[[1]][["raw"]]
                statsResultsRawTest[[pairName]][[targetLen]]  <- partialResults[[2]][["raw"]]
            }
        }
        
        print(statsResults)
        resultsStatsTable <- do.call(rbind.data.frame, statsResults)
        resultsStatsTableRawTrain <- do.call(rbind.data.frame, statsResultsRawTrain)
        resultsStatsTableRawTest  <- do.call(rbind.data.frame, statsResultsRawTest)
        
        
        getF <- function(data, axis, f) 
            apply(data, axis, function(c) f(na.omit(c)))
        
        typesStats <- rbind(
            "Train max" = getF(resultsStatsTableRawTrain, 2, max),
            "Train min" = getF(resultsStatsTableRawTrain, 2, min),
            "Test max" = getF(resultsStatsTableRawTest,  2, max),
            "Test min" = getF(resultsStatsTableRawTest,  2, min)
        )
        
        lenghtsStats <- cbind(
            getF(resultsStatsTableRawTrain, 1, max),
            getF(resultsStatsTableRawTrain, 1, min),
            
            getF(resultsStatsTableRawTest,  1, min),
            getF(resultsStatsTableRawTest,  1, max)
        )
        
        ## save stats table
        resultsStatsTable <- rbind(resultsStatsTable, typesStats)
        fileName <- paste0("results/", dataFileName, "/", dataFileName, "_stats_", reportStatsCategory, ".txt" )
        write.table(resultsStatsTable, fileName, sep = "\t")
        
        ## save implied ranks
        compare <- function(data){
            names <- rownames(data)
            apply(data, 2, function(c){
                indices <- order(c, decreasing = T)
                paste0(names[indices], " (", c[indices], "%)")  
            })
        }
        
        ranksTrain <- compare(resultsStatsTableRawTrain)
        fileName <- paste0("results/", dataFileName, "/", dataFileName, "_ranks_train_", reportStatsCategory, ".txt" )
        write.table(ranksTrain, fileName, sep = "\t")
        
        ranksTest <- compare(resultsStatsTableRawTest)
        fileName <- paste0("results/", dataFileName, "/", dataFileName, "_ranks_test_", reportStatsCategory, ".txt" )
        write.table(ranksTest, fileName, sep = "\t")
    }
}


print("DONE!")

################################################################################
## SELECTION METHODS COMPARE
################################################################################
## Outputs: BOXPLOTS + Stats

sources <- c("train", "test")
targets <- c("accuracy", "accuracy_counts")
titles <- list("accuracy" = "Bag-of-Words", "accuracy_counts" = "Counts")

fileName <- paste0(dataFileName, "_selections_comp.png")
png(fileName, width=8.1, height=5.2, units="in", res=600)
par(mfrow=c(2,2))
statsResults <- c()

for(targetMetricsName in targets){
    for (source in sources){
        results <- list()
        #targetMetricsName <- "accuracy" # accuracy_counts, accuracy
        #source <- "train"
        
        for (dataFileName in dataResultsFileNames){
            data <- fromJSON(dataFileName)
            pairNames <- names(data)
            results[[dataFileName]] <- c()
            
            for(targetLen in existingTargetLens){
                for(pairName in pairNames){
                    values <- as.numeric( data[[pairName]][[targetLen]][[source]][[targetMetricsName]] )
                    values <- na.omit(values)
                    results[[dataFileName]] <- append(results[[dataFileName]], values)
                }
            }
            
            statsResults <- rbind(statsResults, c(paste0(dataFileName, source, targetMetricsName), summary(results[[dataFileName]]) ))
        }
        
        
        d <- melt(results)
        result <- aov(value ~ L1, d)
        summary(result)
        
        t.test( results[[1]], results[[3]])
        
        par(mar=c(2, 4, 2, 1))
        boxplot(
            results, 
            names=c("TOP-50\nfrom corpora","TOP-50\nfrom each text","50 random\nfrom corpora"),
            xlab= "",# Function Words Selection Method",
            ylab="Accuracy",
            ann=T,
            xaxt="n",
            cex=2,
            main= paste0( titles[[targetMetricsName]], " (", source, ")") ,
            bg="black"
        )
        axis(side = 1, at = seq_along(b$names), labels = b$names, tick = FALSE)
    }
}

dev.off()

d <- statsResults[,-1]
d <- as.data.frame( apply(d, 2, as.numeric) )
rownames(d) <- statsResults[,1]

round(statsResults, 3)
