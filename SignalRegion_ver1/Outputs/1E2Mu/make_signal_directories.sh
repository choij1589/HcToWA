#!/bin/sh

# mass points
MHc160=(15 45 75 85 120 155)
MHc130=(15 45 55 90 125)
MHc100=(15 25 60 95)
MHc70=(15 40 65)

for MA in "${MHc160[@]}"
do
  MHc=160
  mkdir MHc${MHc}_MA${MA}
done

for MA in "${MHc130[@]}"
do
  MHc=130
  mkdir MHc${MHc}_MA${MA}
done

for MA in "${MHc100[@]}"
do
  MHc=100
  mkdir MHc${MHc}_MA${MA}
done

for MA in "${MHc70[@]}"
do
  MHc=70
  mkdir MHc${MHc}_MA${MA}
done
