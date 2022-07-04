#!/usr/bin/env bash
# Copyright      2021   Yu-Huai Peng

mfccdir=`pwd`/mfcc
vaddir=`pwd`/mfcc

stage=-1
stop_stage=999

gpu=0

train_data_dir=/mnt/md0/dataset/wsj0_2mix/examples_tr/
dev_data_dir=/mnt/md0/dataset/wsj0_2mix/examples_cv/
eval_data_dir=/mnt/md0/dataset/wsj0_2mix/examples_tt/

train_config=trgspk_filter
checkpoint=none

nj=64
train_cmd="run.pl"
wsj_2mix_scripts="/mnt/md0/user_fal1210/dprnn_attractor/dptnet_xvector/DPT_xvector/create-speaker-mixtures"
wham_wav_dir="/mnt/md0/user_fal1210/corpora/output_mix_clean/wav8k/min"
wsj0_wav_dir='/mnt/md0/user_fal1210/corpora'

. path.sh
. cmd.sh
. parse_options.sh || exit
set -e

if [ $stage -le 0 -a $stop_stage -ge 0 ]; then
    # Path to some, but not all of the training corpora
    local/make_2mix.py $train_data_dir data/train
    local/make_2mix.py $dev_data_dir data/dev
    local/make_2mix.py $eval_data_dir data/test
fi

if [ $stage -le 1 -a $stop_stage -ge 1 ]; then
    for name in train dev test; do
        utils/data/get_reco2dur.sh data/${name}
    done
    make_spk_id.py --write_utt2spk_id true data/train
    make_spk_id.py --spk2spk_id data/train/spk2spk_id --write_utt2spk_id true data/dev
fi

if [ $stage -le 2 -a $stop_stage -ge 2 ]; then
    for name in train dev test; do
        make_melspec.sh data/${name} exp/make_mels/train mels
        ${python_cmd} compute_spec_energy.py --data_type ${name}
    done
fi

if [ ${stage} -le 3 -a ${stop_stage} -ge 3 ]; then
    echo "Stage 3: X-vector Extraction"
    ### Task dependent. You have to make data the following preparation part by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 3-1: Data preparation"
    local/data_prep.sh ${wham_wav_dir} ${wsj_2mix_scripts} \
      ${wsj0_wav_dir}/wsj0 || exit 1;

    echo "stage 3-2: Make MFCC"
    # Make MFCCs and compute the energy-based VAD for each dataset
    mfccdir=mfcc
    vaddir=mfcc
    for name in train dev test wsj0; do
        utils/copy_data_dir.sh data/${name} data/${name}_mfcc_sre16
        #utils/resample_data_dir.sh 16000 data/${name}_mfcc_sre16
        local/make_mfcc.sh \
            --write-utt2num-frames true \
            --mfcc-config conf/mfcc_sre16.conf \
            --nj ${nj} --cmd "$train_cmd" \
            data/${name}_mfcc_sre16 exp/make_mfcc_sre16 ${mfccdir}
        utils/fix_data_dir.sh data/${name}_mfcc_sre16
        sid/compute_vad_decision.sh --nj ${nj} --cmd "$train_cmd" \
            data/${name}_mfcc_sre16 exp/make_vad_sre16 ${vaddir}
        utils/fix_data_dir.sh data/${name}_mfcc_sre16
    done

    echo "stage 3-3: X-vector Extraction"
    # Check pretrained model existence
    nnet_dir=exp/xvector_nnet_1a
    if [ ! -e ${nnet_dir} ]; then
        echo "X-vector model does not exist. Download pre-trained model."
        wget https://kaldi-asr.org/models/3/0003_sre16_v2_1a.tar.gz
        tar xvf 0003_sre16_v2_1a.tar.gz
        mv 0003_sre16_v2_1a/exp/xvector_nnet_1a exp
        rm -rf 0003_sre16_v2_1a.tar.gz 0003_sre16_v2_1a
    fi
    # Extract x-vector
    for name in train dev test wsj0; do
        sid/nnet3/xvector/extract_xvectors.sh --cmd "$train_cmd --mem 4G" --nj ${nj} \
            ${nnet_dir} data/${name}_mfcc_sre16 \
            ${nnet_dir}/xvectors_${name}_sre16
    done
fi

if [ $stage -le 4 -a $stop_stage -ge 4 ]; then
    train.py -c conf/${train_config}.yaml \
        --output_dir exp/${train_config}/ \
        --train_dir data/train \
        --valid_dir data/dev \
        --xvector_dir ${nnet_dir} \
        --gpu ${gpu} \
        --checkpoint $checkpoint
fi

if [ $stage -le 5 -a $stop_stage -ge 5 ]; then
    for name in dev test; do
        evaluate.py -c conf/${train_config}.yaml \
            --output-dir exp/${train_config}/ \
            --eval-dir data/${name} \
            --xvector_file ${nnet_dir}/xvectors_${name}_sre16/spk_xvector.scp
            --gpu ${gpu} \
            --checkpoint ${checkpoint}
        mv exp/${train_config}/eval.log exp/${train_config}/eval_${name}.log
    done
fi
