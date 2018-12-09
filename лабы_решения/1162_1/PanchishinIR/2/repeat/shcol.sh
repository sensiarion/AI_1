#! /usr/bin/env bash

# Show column content
# How to use: shcol.sh <column name>
# Example ./shcol.sh State 

# Dump
# bash -s <

awk -v colname=$1 -F ',' '
BEGIN { idx = 0 }
NR == 1 { 
    for (i = 1; i <= NF; i++) {
        if ($i == colname) {
			idx = i 
		}
    }   
	if ($idx == 0) {
		exit
	}
} 
{ print $idx }
' ../../../../../data/telecom_churn.csv | less
