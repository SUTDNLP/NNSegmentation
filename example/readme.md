#How to run this example?  
`chmod +x run.sh`  
`./run.sh`  
#Result File Description
If everything works as expected, a folder called *pku.sample* will appear in this directory. 
There will be three logs file in *pku.sample* directory. 
They are *SparseCRFMMLabeler.log*, *SparseLSTMCRFMMLabeler.log* and *LSTMCRFMMLabeler.log*.
Each log file records the performance on the dev set and test set. 

You can use 
`grep Exceed logfile -C 4` 
to see the performance.  

For example, performing `grep Exeed SparseCRFMMLabeler.log -C 4` will show similar messages below  
`Recall:P=43285/46549=0.92988, Accuracy:P=43285/46435=0.932163, Fmeasure:0.93102`    

`test:`  

`Recall:P=83800/90886=0.922034, Accuracy:P=83800/90310=0.927915, Fmeasure:0.924965 `   

`Exceeds best previous performance of 0.9309. Saving model file..`    

The first "Recall..." line shows you the performance of the dev set and the second "Recall..." line shows 
you the performance of the test set.   

Also there will be three directories produced inside *pku.sample*. 
These directories are *SparseCRFMMLabeler*, *LSTMCRFMMLabeler* and *SparseLSTMCRFMMLabeler*.
Inside each directory, there are two files *pku.dev.featsOUTnodrop* and *pku.test.featsOUTnodrop*, which are corresponding to the best tagged result of dev set and test set respectively until now.
.
#Feature Template
+ character unigram,  Ci\_i  ( -2=<i<=2 ). 
+ character bigram,  C\_{i-1}C\_i   ( -2=<i<2 ),  C-1C1, C0C2
+ whether two characters are equal, RC0C-2 and RC0C-1
+ character trigram, C-1C0C1
+ type(C0),  there are five types.  0: Punctuation, 1: Alphabet, 2:Date, 3: Number, 4: others
+ type(C-1C0C1)
+ type(C-2C-1C0C1C2)

For example, considering this sentence 
`共同  创造 美好  的  新  世纪  ——  二○○一年  新年  贺词`, the extracted features for the fifth character "美" is   
`美 [T1]造美 [T2]创造美 [S]C-2=创 [S]C-1=造 [S]C0=美 [S]C1=好 [S]C2=的 [S]C-2C-1=创造 [S]C-1C0=造美 [S]C0C1=美好 [S]C1C2=好的 [S]C-1C1=造好 [S]C0C2=美的 [S]RC0C-2=0 [S]RC0C-1=0 [S]C-1C0C1=造美好 [S]TC-1=4 [S]TC-11==444 [S]TC-22==44444 b-seg`   
where
* 美 is the current character. You should use "-word" to specify the character unigram embeddings.
* [T1] and [T2]. Things started with "[T" are additional targets which need to be embedded. Here we use character bigram embeddings and character trigram embeddings.  You should use "-tag" to specify these embeddings and use comma as a delimiter between embedding file paths. 
For example, "-tag t2.vec,t3.vec".
* [S]. Things startd with [S] are sparse features. 
* b-seg is the tag for current character. Tags must be augmented with '-seg' postfix to indicate this is a segmentation task but not a classification application.

#How to use more embeddings?
First, you should add a item started with "[T" such as "[T3]" to your feature file.  
Second, you need to provide the embedding file using the "-tag" command option. 


