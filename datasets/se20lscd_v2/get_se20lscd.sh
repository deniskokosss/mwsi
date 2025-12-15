#!/bin/bash
wget https://zenodo.org/record/5796878/files/dwug_en.zip
wget https://zenodo.org/record/5796871/files/dwug_de.zip
wget https://zenodo.org/record/5801358/files/dwug_sv.zip
#wget https://zenodo.org/record/5255228/files/dwug_la.zip  # Latin was annotated differently from other languages, skip it
echo Unzipping...
for x in *zip; do unzip -q $x; done

for p in "" opt sense; do 
  for l in en de sv; do
    if [ "$p" = sense  ] && [ "$l" != de ]
    then
        continue
    fi
 
    mkdir -p $l
    echo Converting $l $p
    python convert.py  ${l} ${p}
  done
done
