export KALDI_ROOT=/mnt/md0/user_fal1210/SLAM/kaldi
[ ! -d $PWD/steps ] && ln -s $KALDI_ROOT/egs/wsj/s5/steps steps
[ ! -d $PWD/utils ] && ln -s $KALDI_ROOT/egs/wsj/s5/utils utils
[ ! -d $PWD/sid ] && ln -s $KALDI_ROOT/egs/sre08/v1/sid
[ ! -d $PWD/diarization ] && ln -s $KALDI_ROOT/egs/callhome_diarization/v1/diarization
export PATH=$PWD/utils/:$KALDI_ROOT/tools/openfst/bin:$KALDI_ROOT/tools/sph2pipe_v2.5:$KALDI_ROOT/tools/sctk/bin:$PWD:$PATH
[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
. $KALDI_ROOT/tools/config/common_path.sh
export PATH=../../../utt2spks/bin:../../../utils:$PATH
export PYTHONPATH=../../..:$PYTHONPATH
