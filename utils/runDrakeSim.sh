#! /bin/sh
if cd $DRAKE_PATH_ROOT; then
	bazel run @drake//examples/kuka_iiwa_arm:kuka_simulation
else
	echo "You need to set DRAKE_PATH_ROOT to your drake path"
fi
