#! /bin/sh
if cd $DRAKE_PATH_ROOT; then
	bazel run @drake//lcmtypes:drake-lcm-spy
else
	echo "You need to set DRAKE_PATH_ROOT to your drake path"
fi
