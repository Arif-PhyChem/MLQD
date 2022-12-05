if grep -q "FINAL ML MODEL CREATED AND SAVED" $PWD/kkr_train_output; then 
	echo "grep.bash: MLatom execution was successful, Cheers!"
else
	echo "grep.bash: MLatom execution was not successful, Check your input files etc. !"
fi


