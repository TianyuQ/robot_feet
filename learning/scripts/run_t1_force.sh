#!/bin/bash
# T1 Force Humanoid Robot - Training and Evaluation Script
# This script provides convenient commands for training and evaluating the T1_Force policy

set -e  # Exit on error

# Configuration
CONDA_ENV="robot_feet_311"
SCRIPT_DIR="/home/tq877/Tianyu/robot_feet/robot_feet/learning/scripts"
LOGS_DIR="/home/tq877/Tianyu/robot_feet/robot_feet/logs"
CHECKPOINT_PATH="${LOGS_DIR}/T1ForceJoystickFlatTerrain-20251202_235023/checkpoints/"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored messages
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Function to activate conda environment
activate_env() {
    print_info "Activating conda environment: ${CONDA_ENV}"
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate ${CONDA_ENV}
    if [ $? -ne 0 ]; then
        print_warning "Failed to activate conda environment. Trying alternative method..."
        eval "$(conda shell.bash hook)"
        conda activate ${CONDA_ENV}
    fi
    print_success "Conda environment activated"
}

# Function to change to script directory
cd_script_dir() {
    print_info "Changing to script directory: ${SCRIPT_DIR}"
    cd ${SCRIPT_DIR}
}

# Function to find latest checkpoint
find_latest_checkpoint() {
    local exp_dir=$1
    if [ -d "${exp_dir}/checkpoints" ]; then
        local latest=$(ls -1 "${exp_dir}/checkpoints" | grep "^000" | sort -n | tail -1)
        if [ -n "$latest" ]; then
            echo "${exp_dir}/checkpoints/${latest}"
        else
            echo "${exp_dir}/checkpoints"
        fi
    else
        echo ""
    fi
}

# Main menu
show_menu() {
    echo ""
    echo "=========================================="
    echo "  T1 Force Training & Evaluation Menu"
    echo "=========================================="
    echo "1. Train new policy"
    echo "2. Train with custom force_impulse_penalty scale"
    echo "3. Evaluate policy (basic)"
    echo "4. Evaluate policy with plots"
    echo "5. Noise evaluation (test if policy is quieter)"
    echo "6. Noise evaluation with custom trajectories"
    echo "7. Evaluate with custom velocity commands"
    echo "8. Interactive command testing"
    echo "9. Run all evaluations"
    echo "10. Exit"
    echo ""
    echo -n "Select an option [1-10]: "
}

# Training function
train_policy() {
    local force_penalty_scale=${1:-""}
    print_info "Starting policy training..."
    activate_env
    cd_script_dir
    
    print_info "Training will save checkpoints to: ${LOGS_DIR}/T1ForceJoystickFlatTerrain-<timestamp>/checkpoints/"
    
    if [ -n "$force_penalty_scale" ]; then
        print_info "Using force_impulse_penalty scale: ${force_penalty_scale}"
        python t1_joystick_flat_terrain_force.py --train --force-impulse-penalty-scale ${force_penalty_scale}
    else
        python t1_joystick_flat_terrain_force.py --train
    fi
    
    print_success "Training completed!"
    print_info "Checkpoints saved in logs directory"
}

# Basic evaluation
eval_policy() {
    local checkpoint=$1
    if [ -z "$checkpoint" ]; then
        checkpoint=$CHECKPOINT_PATH
    fi
    
    print_info "Evaluating policy from: ${checkpoint}"
    activate_env
    cd_script_dir
    
    python t1_joystick_flat_terrain_force.py --eval \
        --load-checkpoint-path "${checkpoint}"
    
    print_success "Evaluation completed!"
}

# Evaluation with plots
eval_policy_with_plots() {
    local checkpoint=$1
    if [ -z "$checkpoint" ]; then
        checkpoint=$CHECKPOINT_PATH
    fi
    
    print_info "Evaluating policy with plots from: ${checkpoint}"
    activate_env
    cd_script_dir
    
    python t1_joystick_flat_terrain_force.py --eval --plot \
        --load-checkpoint-path "${checkpoint}"
    
    print_success "Evaluation with plots completed!"
    print_info "Plots and videos saved in the experiment directory"
}

# Noise evaluation
noise_eval() {
    local checkpoint=$1
    local num_trajectories=${2:-10}
    
    if [ -z "$checkpoint" ]; then
        checkpoint=$CHECKPOINT_PATH
    fi
    
    print_info "Running noise evaluation with ${num_trajectories} trajectories"
    print_info "Checkpoint: ${checkpoint}"
    activate_env
    cd_script_dir
    
    python t1_joystick_flat_terrain_force.py --noise-eval \
        --num-trajectories ${num_trajectories} \
        --load-checkpoint-path "${checkpoint}"
    
    print_success "Noise evaluation completed!"
}

# Evaluate with custom commands
eval_custom_commands() {
    local checkpoint=$1
    local x_vel=${2:-0.0}
    local y_vel=${3:-0.0}
    local yaw_vel=${4:-3.14}
    
    if [ -z "$checkpoint" ]; then
        checkpoint=$CHECKPOINT_PATH
    fi
    
    print_info "Evaluating with custom commands: Vx=${x_vel}, Vy=${y_vel}, Vyaw=${yaw_vel}"
    activate_env
    cd_script_dir
    
    python t1_joystick_flat_terrain_force.py --eval --plot \
        --x-vel ${x_vel} --y-vel ${y_vel} --yaw-vel ${yaw_vel} \
        --load-checkpoint-path "${checkpoint}"
    
    print_success "Custom command evaluation completed!"
}

# Interactive testing
interactive_test() {
    local checkpoint=$1
    if [ -z "$checkpoint" ]; then
        checkpoint=$CHECKPOINT_PATH
    fi
    
    print_info "Starting interactive command testing"
    activate_env
    cd_script_dir
    
    python t1_joystick_flat_terrain_force.py --interactive \
        --load-checkpoint-path "${checkpoint}"
}

# Run all evaluations
run_all_evals() {
    local checkpoint=$1
    if [ -z "$checkpoint" ]; then
        checkpoint=$CHECKPOINT_PATH
    fi
    
    print_info "Running all evaluations..."
    eval_policy_with_plots "$checkpoint"
    echo ""
    noise_eval "$checkpoint" 10
    print_success "All evaluations completed!"
}

# Main execution
main() {
    # Check if argument provided (for direct command execution)
    if [ $# -gt 0 ]; then
        case $1 in
            train)
                train_policy "$2"
                ;;
            eval)
                eval_policy "$2"
                ;;
            eval-plot)
                eval_policy_with_plots "$2"
                ;;
            noise)
                noise_eval "$2" "$3"
                ;;
            custom)
                eval_custom_commands "$2" "$3" "$4" "$5"
                ;;
            interactive)
                interactive_test "$2"
                ;;
            all)
                run_all_evals "$2"
                ;;
            *)
                echo "Usage: $0 {train|eval|eval-plot|noise|custom|interactive|all} [checkpoint_path] [additional_args]"
                echo ""
                echo "Examples:"
                echo "  $0 train"
                echo "  $0 eval"
                echo "  $0 eval-plot"
                echo "  $0 noise 20"
                echo "  $0 custom \"\" 1.0 0.5 2.0"
                echo "  $0 all"
                exit 1
                ;;
        esac
        exit 0
    fi
    
    # Interactive menu mode
    while true; do
        show_menu
        read choice
        
        case $choice in
            1)
                train_policy
                ;;
            2)
                echo -n "Enter force_impulse_penalty scale (default -0.1, use 0 to disable): "
                read force_scale
                force_scale=${force_scale:--0.1}
                train_policy "$force_scale"
                ;;
            3)
                echo -n "Enter checkpoint path (or press Enter for default): "
                read custom_checkpoint
                eval_policy "$custom_checkpoint"
                ;;
            4)
                echo -n "Enter checkpoint path (or press Enter for default): "
                read custom_checkpoint
                eval_policy_with_plots "$custom_checkpoint"
                ;;
            5)
                echo -n "Enter checkpoint path (or press Enter for default): "
                read custom_checkpoint
                noise_eval "$custom_checkpoint" 10
                ;;
            6)
                echo -n "Enter checkpoint path (or press Enter for default): "
                read custom_checkpoint
                echo -n "Enter number of trajectories (default 10): "
                read num_traj
                num_traj=${num_traj:-10}
                noise_eval "$custom_checkpoint" "$num_traj"
                ;;
            7)
                echo -n "Enter checkpoint path (or press Enter for default): "
                read custom_checkpoint
                echo -n "Enter X velocity (default 0.0): "
                read x_vel
                echo -n "Enter Y velocity (default 0.0): "
                read y_vel
                echo -n "Enter Yaw velocity (default 3.14): "
                read yaw_vel
                x_vel=${x_vel:-0.0}
                y_vel=${y_vel:-0.0}
                yaw_vel=${yaw_vel:-3.14}
                eval_custom_commands "$custom_checkpoint" "$x_vel" "$y_vel" "$yaw_vel"
                ;;
            8)
                echo -n "Enter checkpoint path (or press Enter for default): "
                read custom_checkpoint
                interactive_test "$custom_checkpoint"
                ;;
            9)
                echo -n "Enter checkpoint path (or press Enter for default): "
                read custom_checkpoint
                run_all_evals "$custom_checkpoint"
                ;;
            10)
                print_info "Exiting..."
                exit 0
                ;;
            *)
                print_warning "Invalid option. Please select 1-10."
                ;;
        esac
        
        echo ""
        echo "Press Enter to continue..."
        read
    done
}

# Run main function
main "$@"

