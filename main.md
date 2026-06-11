Using workflow from https://link.springer.com/article/10.1186/s12886-021-01830-9

I copied EAS, EUR, and KAZ binary files + final_annovared.tsv into folder paper2_akilzhanova
In the paper they used 138 SNPs. I will also use a a certain number of SNPs using cardio panel. I saved the file trusight into genes.tsv (just keeping the gene names)

cat final_annovared_extended.tsv | cut -f 1,2,4,5,7,24,456 > selected_annotation.tsv
grep -Fw -f genes.tsv selected_annotation.tsv > only_cardio.tsv
awk '$5 !~ /[;-]/' only_cardio.tsv > only_cardio2.tsv

I have 8194 SNPs with genes that match the panel; some matches are parial so i will remove all genes where there is a ; or -

cat only_cardio2.tsv | cut -f 5 | sed 's/;.*//' | sort | uniq | wc -l
157/174 genes in the list are represented in my snp list of 4262 

cat only_cardio2.tsv | cut -f 5 | sort | uniq | sort > list1.txt
cat genes.tsv | sort > list2.txt
diff list1.txt list2.txt 

All good, I only have genes that are in the list now in only_cardio2.tsv

