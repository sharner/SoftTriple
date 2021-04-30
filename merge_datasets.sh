#!/usr/bin/env bash

count=1
for fn in $1/train/* $1/test/* $2/train/* $2/test/*
do
  echo $fn
  echo $3/$count
  cp -r $fn $3/$count
  ((count++))
done
