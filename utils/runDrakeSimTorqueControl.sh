#! /bin/sh
if [ -z "$DRAKE_PATH_ROOT"]; then
	echo "\n[!] ERROR: You need to put in you .bashrc export DRAKE_PATH_ROOT=<path_to_drake>\n"
else
	cd $DRAKE_PATH_ROOT;
	bazel run @drake//examples/kuka_iiwa_arm:kuka_simulation -- --torque_control;
fi