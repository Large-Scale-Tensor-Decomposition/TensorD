export PYTHON_PATH=/root/tensorD:$PYTHON_PATH


run_cp_ex() {
    N1=$1
    N2=$2
    N3=$3
    gR=$4
    dR=$5
    time=0
    while [ "$time" != 20 ]
    do
        time=$(($time+1))
        python3 cp_ex.py $N1 $N2 $N3 $gR $dR $time
    done
}

run_ncp_ex() {
    N1=$1
    N2=$2
    N3=$3
    gR=$4
    dR=$5
    time=0
    while [ "$time" != 20 ]
    do
        time=$(($time+1))
        python3 ncp_ex.py $N1 $N2 $N3 $gR $dR $time
    done
}

run_tucker_ex() {
    N1=$1
    N2=$2
    N3=$3
    gR=$4
    dR=$5
    time=0
    while [ "$time" != 20 ]
    do
        time=$(($time+1))
        python3 tucker_ex.py $N1 $N2 $N3 $gR $dR $time
    done
}

run_ntucker_ex() {
    N1=$1
    N2=$2
    N3=$3
    gR=$4
    dR=$5
    time=0
    while [ "$time" != 20 ]
    do
        time=$(($time+1))
        python3 ntucker_ex.py $N1 $N2 $N3 $gR $dR $time
    done
}

N=20
while [ "$N" != 320 ]
do
    gR=10
    dR=5
    while [ "$dR" != 20 ]
    do
        run_cp_ex $N $N $N $gR $dR
        run_ncp_ex $N $N $N $gR $dR
        run_tucker_ex $N $N $N $gR $dR
        run_ntucker_ex $N $N $N $gR $dR
        dR=$(($dR+5))
    done

    gR=20
    dR=15
    while [ "$dR" != 30 ]
    do
        run_cp_ex $N $N $N $gR $dR
        run_ncp_ex $N $N $N $gR $dR
        run_tucker_ex $N $N $N $gR $dR
        run_ntucker_ex $N $N $N $gR $dR
        dR=$(($dR+5))
    done

    N=$(($N*2))
done
















