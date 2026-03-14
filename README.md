# DeepOCR                                          
## 預處理:

### 1.positive
python src/fix_bed_all_chrom.py --bed data/cattle/Rumen/pos.bed --out data/cattle/Rumen/pos.fixed.bed
羊才要:
python src/fix_cattle_bed.py data/cattle/Rumen/pos.bed data/cattle/Rumen/pos.fixed.bed

### 2.negative
python src/fix_bed_all_chrom.py --bed data/cattle/Rumen/neg.bed --out data/cattle/Rumen/neg.fixed.bed
羊才要:
python src/fix_cattle_bed.py data/cattle/Rumen/neg.bed data/cattle/Rumen/neg.fixed.bed

### 3.轉檔
python src/bed_to_fasta.py --bed data/cattle/Rumen/pos.fixed.bed --ref data/cattle/Rumen/genome.fa --out data/cattle/Rumen/pos.fa
python src/bed_to_fasta.py --bed data/cattle/Rumen/neg.fixed.bed --ref data/cattle/Rumen/genome.fa --out data/cattle/Rumen/neg.fa



## 資料切割:
###  1.前1000
python src/preprocess.py --out data/cattle/Rumen/output/ --pos data/cattle/Rumen/pos.fa --neg data/cattle/Rumen/neg.fa --mode first1000

### 2.center
python src/preprocess.py --pos data/cattle/Rumen/pos.fa --neg data/cattle/Rumen/neg.fa --out data/cattle/Rumen/output/ --mode center --window 1000

### 3.window
python src/preprocess.py --pos data/cattle/Rumen/pos.fa --neg data/cattle/Rumen/neg.fa --out data/cattle/Rumen/output/ --mode window --window 1000 --step 200

### 4.smooth
python src/preprocess.py --pos data/cattle/Rumen/pos.fa --neg data/cattle/Rumen/neg.fa --out data/cattle/Rumen/output/ --mode smooth --window 1000

### 5.搭配k-mer和local(推薦)
(1)前1000+local4+3-mer
python src/preprocess.py --out data/cattle/Rumen/output/ --pos data/cattle/Rumen/pos.fa --neg data/cattle/Rumen/neg.fa --mode first1000 --local_seg 4 --k 3
(2)前1000+local5+4-mer
python src/preprocess.py --out data/cattle/Rumen/output/ --pos data/cattle/Rumen/pos.fa --neg data/cattle/Rumen/neg.fa --mode first1000 --local_seg 5 --k 4



## 訓練:(根據需要的方法靈活調整)(以下示範幾種)
### 1.base
python src/train.py --seq data/cattle/Rumen/output/data_onehot.npy --label data/cattle/Rumen/output/label.npy --out data/cattle/Rumen/model/  --val 0.1 --random 0.2

### 2.base+local4
python src/train.py --seq data/cattle/Rumen/output/data_onehot.npy --local data/cattle/Rumen/output/local_seg4.npy --label data/cattle/Rumen/output/label.npy --out data/cattle/Rumen/model/ --val 0.1 --random 0.2

### 3.base+3mer
python src/train.py --seq data/cattle/Rumen/output/data_onehot.npy --kmer data/cattle/Rumen/output/kmer_k3.npy --label data/cattle/Rumen/output/label.npy --out data/cattle/Rumen/model/ --val 0.1 --random 0.2

### 4.base+3mer+local4
python src/train.py --seq data/cattle/Rumen/output/data_onehot.npy --local data/cattle/Rumen/output/local_seg4.npy --kmer data/cattle/Rumen/output/kmer_k3.npy --label data/cattle/Rumen/output/label.npy --out data/cattle/Rumen/model/ --val 0.1 --random 0.2



## 預測
python src/predict.py --model data/cattle/Rumen/model/model_random.keras --seq data/cattle/Rumen/pos.fa --out data/cattle/Rumen/output/



## 畫圖
python src/motif_visualize.py --model data/cattle/Rumen/model/model_random.keras --out data/cattle/Rumen/output/motif/ 
