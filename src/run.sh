# 20201.01.28
# @yifan
# use the jpeg codec to compress the coefficients
#
folder="tmp"
output="/Users/alex/Documents/GitHub/PCS/result/""$folder""/"

for (( i=0; i <24 ; i+=1 ))
do
    for ((N=0; N <25; N+=1))
    do
        name="$i"_"$N"
        cp /Users/alex/Documents/GitHub/PCS/result/"$folder"/"$name".txt ~/desktop/tmp.txt
        ./jpeg-6b/cjpeg  -quality 99 -outfile "$output""$name".jpg  /Users/alex/Desktop/proj/compression/data/Kodak/Kodak/"$i".bmp 
        echo Finish image $i @Qf=$N
    done
done

