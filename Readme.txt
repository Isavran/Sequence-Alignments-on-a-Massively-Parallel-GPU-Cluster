Large-scale-Pairwise-Sequence-Alignments-on-a-Massively-Parallel-GPU-Cluster

Next Generation Sequencing (NGS) technologies use parallel sequencing technology 
to generate a large number of short DNA reads in a single run, allowing for de 
novo assembly of whole genomes. 

One of the key operations needed when performing analysis and de novo assembly 
is to cluster the sequences based on their similarity, where the sequence 
similarity is determined using a global alignment algorithm. Even tough 
individual alignments are relatively inexpensive, yet computing the pairwise 
distances requires O(n^2) alignments, which becomes prohibitively expensive for 
large datasets. 

We present our updated kernel, in which we solved a crucial performance 
limitation related to memory usage, and our results from scaling to Stampede, 
a massive GPU cluster. By executing our kernel on 32 of Stampede’s NVIDIA K20 
GPU.
