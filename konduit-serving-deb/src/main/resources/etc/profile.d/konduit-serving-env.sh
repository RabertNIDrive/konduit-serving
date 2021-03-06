#!/usr/bin/env bash
# This file defines the environment variables needed to run the konduit serving admin command.

export KONDUIT_SERVING_BASE_ARTIFACT_NAME=konduit-serving
export KONDUIT_SERVING_DIRECTORY_STRUCTURE=konduit/serving
export KONDUIT_SERVING_SPIN_VERSION=custom
export KONDUIT_SERVING_VERSION=0.1.0-SNAPSHOT
export KONDUIT_SERVING_BACKEND=cpu

export KONDUIT_SERVING_HOME=/opt/${KONDUIT_SERVING_DIRECTORY_STRUCTURE}
export CONDA_INSTALL_PATH=${KONDUIT_SERVING_HOME}/miniconda
export KONDUIT_SERVING_LOG_DIR=/var/log/${KONDUIT_SERVING_DIRECTORY_STRUCTURE}
export KONDUIT_JAR_PATH=${KONDUIT_SERVING_HOME}/${KONDUIT_SERVING_BASE_ARTIFACT_NAME}-${KONDUIT_SERVING_VERSION}-${KONDUIT_SERVING_SPIN_VERSION}-${KONDUIT_SERVING_BACKEND}.jar
export KONDUIT_SERVING_PID_FILE=${KONDUIT_SERVING_HOME}/${KONDUIT_SERVING_BASE_ARTIFACT_NAME}.pid
export KONDUIT_SERVING_BIN_DIRECTORY=${KONDUIT_SERVING_HOME}/bin