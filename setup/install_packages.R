#!/usr/bin/env Rscript

# vector of CRAN packages to be installed
cran_packages <- c("ggplot2",
"GGally",
"ggdendro",
"RColorBrewer",
"dendextend",
"gplots",
"doMC",
"dbscan",
"cluster",
"methods",
"devtools",
"caret",
"mlbench",
"plot3D",
"pROC",
"ROCR",
"Rtsne",
"scatterplot3d",
"UsingR",
"Rafalib",
"downloader",
"lattice",
"e1071"
)

# vector of bioconductor packages to be installed
bioc_packages <- c("EBImage")

# install CRAN packages
install.packages(cran_packages, repos='http://mirrors.ebi.ac.uk/CRAN/')

# install bioconductor packages
source("https://bioconductor.org/biocLite.R")
biocLite(bioc_packages)

# install packages from other repositories
devtools::install_github("SheffieldML/vargplvm/vargplvmR")
devtools::install_github("ManchesterBioinference/DEtime")




