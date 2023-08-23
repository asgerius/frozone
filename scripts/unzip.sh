# This script recursively unzips every .zip in DATAPATH

DATAPATH=data

for zipfile in $(ls -d $(find $DATAPATH) | grep .zip)
do
    echo $zipfile
    unzip -o $zipfile -d $(dirname $zipfile) > /dev/null
done
