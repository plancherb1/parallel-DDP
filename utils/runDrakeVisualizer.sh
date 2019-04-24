#! /bin/sh
if cd $DRAKE_PATH_ROOT; then
	bazel run @drake//tools:drake_visualizer
else
	echo "You need to set DRAKE_PATH_ROOT to your drake path"
fi
