#!/bin/bash

function scan_attacks () {
    if [[ $# -lt 3 ]]; then
        echo "Usage: scan_attacks model_path config_path result_dir_path"
        return 2
    fi
    model=$1
    config=$2
    output_dir=$3

    mkdir -p ${output_dir}
    semcmd="sem -j $(($(nproc --all)/2))"

    $semcmd python test.py --config-path "$config" --load-model "$model" ">" "${output_dir}/clean.log"
    $semcmd python test.py --config-path "$config" --load-model "$model" --deterministic ">" "${output_dir}/clean_deterministic.log"
    $semcmd python test.py --config-path "$config" --load-model "$model" --attack-method action ">" "${output_dir}/mad_attack.log"
    $semcmd python test.py --config-path "$config" --load-model "$model" --attack-method action --deterministic ">" "${output_dir}/mad_attack_deterministic.log"
    $semcmd python test.py --config-path "$config" --load-model "$model" --attack-method critic ">" "${output_dir}/critic_attack.log"
    $semcmd python test.py --config-path "$config" --load-model "$model" --attack-method critic --deterministic ">" "${output_dir}/critic_attack_deterministic.log"
    for sarsa_eps in 0.02 0.05 0.1 0.15 0.2 0.3; do
        for sarsa_reg in 0.1 0.3 1.0 3.0 10.0; do
            echo ${sarsa_eps} ${sarsa_reg}
            $semcmd \
            python test.py --config-path "$config" --load-model "$model" --sarsa-model-path "${output_dir}/sarsa_${sarsa_eps}_${sarsa_reg}.model" --sarsa-enable --sarsa-eps ${sarsa_eps} --sarsa-reg ${sarsa_reg} ">" "${output_dir}/sarsa_${sarsa_eps}_${sarsa_reg}_train.log" ";" \
            python test.py --config-path "$config" --load-model "$model" --attack-sarsa-network "${output_dir}/sarsa_${sarsa_eps}_${sarsa_reg}.model" --attack-method sarsa ">" "${output_dir}/sarsa_${sarsa_eps}_${sarsa_reg}_attack.log" ";" \
            python test.py --config-path "$config" --load-model "$model" --attack-sarsa-network "${output_dir}/sarsa_${sarsa_eps}_${sarsa_reg}.model" --attack-method sarsa --deterministic ">" "${output_dir}/sarsa_${sarsa_eps}_${sarsa_reg}_attack_deterministic.log"
        done
    done
    sem --wait

    echo "clean reward:"
    tail -n1 "${output_dir}/clean.log"
    echo "clean reward (deterministic):"
    tail -n1 "${output_dir}/clean_deterministic.log"
    echo "critic attack reward:"
    tail -n1 "${output_dir}/critic_attack.log"
    echo "critic attack reward (deterministic):"
    tail -n1 "${output_dir}/critic_attack_deterministic.log"
    echo "MAD attack reward:"
    tail -n1 "${output_dir}/mad_attack.log"
    echo "MAD attack reward (deterministic):"
    tail -n1 "${output_dir}/mad_attack_deterministic.log"
    echo "minimum RS attack reward:" 
    (for i in ${output_dir}/*_attack.log; do tail -n1 $i | tr -d ',' | cut -d' ' -f 2; done) | sort -h | head -n 1
    echo "minimum RS attack reward (deterministic action):" 
    (for i in ${output_dir}/*_attack_deterministic.log; do tail -n1 $i | tr -d ',' | cut -d' ' -f 2; done) | sort -h | head -n 1
}

# Example:
# scan_attacks models/model-sappo-convex-humanoid.model config_humanoid_robust_ppo_convex.json sarsa_humanoid_sappo-convex
