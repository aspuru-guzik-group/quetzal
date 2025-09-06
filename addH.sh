#!/bin/bash
start_time=$(date +%s)

mkdir -p samples/hdeco/obabel_hydride
for i in {0..13082}
do
    obabel samples/hdeco/no_H/i$i.xyz -O temp.mol
    hydride -i temp.mol -o tempH.mol
    obabel tempH.mol -O samples/hdeco/obabel_hydride/o$i.xyz
done

end_time=$(date +%s)
elapsed_time=$((end_time - start_time))

echo "Elapsed time: $elapsed_time seconds"
