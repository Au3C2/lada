FROM python:3.13.12-slim-trixie as build
RUN useradd --create-home lada
RUN apt-get update \
    && apt-get install -y --no-install-recommends gettext patch curl xz-utils \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
USER lada
WORKDIR /home/lada
RUN mkdir -p .local/bin \
    && curl -L 'https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-linux64-gpl.tar.xz' \
        | tar xJf - -C .local/bin --strip-components 2 ffmpeg-master-latest-linux64-gpl/bin/ffmpeg ffmpeg-master-latest-linux64-gpl/bin/ffprobe \
    && chmod 555 .local/bin/*
COPY packaging/docker/requirements.txt packaging/docker/requirements.txt
RUN pip install --user --no-deps --no-cache-dir --requirement packaging/docker/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128
COPY patches patches
RUN    patch -u -p1 -d .local/lib/python3.13/site-packages < patches/increase_mms_time_limit.patch \
    && patch -u -p1 -d .local/lib/python3.13/site-packages < patches/remove_ultralytics_telemetry.patch \
    && patch -u -p1 -d .local/lib/python3.13/site-packages < patches/fix_loading_mmengine_weights_on_torch26_and_higher.diff
COPY --chown=lada:lada . .
RUN sh translations/compile_po.sh --release
RUN pip install --user --no-deps --no-cache-dir  '.'
# Download the full model weight set (checksums-verified) so the image is self-contained.
# The weights stay read-only so a compromised runtime cannot tamper with them.
RUN python3 packaging/download_model_weights.py && chmod -R a-w model_weights


FROM python:3.13.12-slim-trixie
RUN useradd --create-home lada
RUN mkdir -p /home/lada/.config/Ultralytics \
    && echo '{"settings_version":"0.0.6","datasets_dir":"datasets","weights_dir":"weights","runs_dir":"experiments","uuid":"dummy","sync":false,"api_key":"","openai_api_key":"","clearml":false,"comet":false,"dvc":false,"hub":false,"mlflow":false,"neptune":false,"raytune":false,"tensorboard":false,"wandb":false,"vscode_msg":false,"openvino_msg":false}' > /home/lada/.config/Ultralytics/settings.json \
    && chown -R lada:lada /home/lada/.config
RUN mkdir -p mnt && chmod 777 mnt
COPY --from=build --chown=lada:lada /home/lada/model_weights /model_weights
COPY --from=build --chown=lada:lada /home/lada/.local /home/lada/.local
USER lada
ENV PATH=/home/lada/.local/bin:$PATH
ENV LOCALE_DIR=/home/lada/.local/lib/python3.13/site-packages/lada/locale

ENTRYPOINT ["lada-cli"]
CMD ["--help"]