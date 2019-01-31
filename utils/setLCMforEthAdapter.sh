export LCM_DEFAULT_URL=udpm://239.255.76.67:7667?ttl=1
if [ `ip route show 224.0.0.0/4 | wc -l` -eq 0 ]
then
    sudo route add -net 224.0.0.0 netmask 240.0.0.0 dev enx70886b8026a5
    # sudo route add -net 224.0.0.0 netmask 240.0.0.0 dev enx70886b80a182
fi