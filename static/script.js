document.addEventListener('DOMContentLoaded', () => {
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('audio-file-input');
    const filePreviewBar = document.getElementById('file-preview-bar');
    const selectedFilename = document.getElementById('selected-filename');
    const btnClearFile = document.getElementById('btn-clear-file');
    const audioPlayerWrapper = document.getElementById('audio-player-wrapper');
    const audioPlayer = document.getElementById('audio-player');
    const btnAnalyze = document.getElementById('btn-analyze');
    const analyzeSpinner = document.getElementById('analyze-spinner');
    
    const emptyState = document.getElementById('empty-state');
    const analysisResults = document.getElementById('analysis-results');

    const predictionBanner = document.getElementById('prediction-banner');
    const predBadge = document.getElementById('pred-badge');
    const predConf = document.getElementById('pred-conf');
    const predTitle = document.getElementById('pred-title');
    const predDesc = document.getElementById('pred-desc');

    const valNormal = document.getElementById('val-normal');
    const barNormal = document.getElementById('bar-normal');
    const valMurmur = document.getElementById('val-murmur');
    const barMurmur = document.getElementById('bar-murmur');
    const valArtifact = document.getElementById('val-artifact');
    const barArtifact = document.getElementById('bar-artifact');

    const imgWaveform = document.getElementById('img-waveform');
    const imgSpectrogram = document.getElementById('img-spectrogram');

    // Tab Mode Switching
    const tabFileMode = document.getElementById('tab-file-mode');
    const tabMicMode = document.getElementById('tab-mic-mode');
    const tabEsp32Mode = document.getElementById('tab-esp32-mode');
    const fileModeContainer = document.getElementById('file-mode-container');
    const micModeContainer = document.getElementById('mic-mode-container');
    const esp32ModeContainer = document.getElementById('esp32-mode-container');

    if (tabFileMode && tabMicMode && tabEsp32Mode) {
        tabFileMode.addEventListener('click', () => {
            tabFileMode.classList.add('active');
            tabMicMode.classList.remove('active');
            tabEsp32Mode.classList.remove('active');
            if (fileModeContainer) fileModeContainer.classList.remove('hidden');
            if (micModeContainer) micModeContainer.classList.add('hidden');
            if (esp32ModeContainer) esp32ModeContainer.classList.add('hidden');
        });

        tabMicMode.addEventListener('click', () => {
            tabMicMode.classList.add('active');
            tabFileMode.classList.remove('active');
            tabEsp32Mode.classList.remove('active');
            if (micModeContainer) micModeContainer.classList.remove('hidden');
            if (fileModeContainer) fileModeContainer.classList.add('hidden');
            if (esp32ModeContainer) esp32ModeContainer.classList.add('hidden');
        });

        tabEsp32Mode.addEventListener('click', () => {
            tabEsp32Mode.classList.add('active');
            tabFileMode.classList.remove('active');
            tabMicMode.classList.remove('active');
            if (esp32ModeContainer) esp32ModeContainer.classList.remove('hidden');
            if (fileModeContainer) fileModeContainer.classList.add('hidden');
            if (micModeContainer) micModeContainer.classList.add('hidden');
        });
    }

    // Auto-poll for ESP32 incoming transmissions every 3 seconds
    let lastEsp32Timestamp = null;
    setInterval(async () => {
        try {
            const resp = await fetch('/api/esp32/latest');
            const data = await resp.json();
            if (data && data.status === 'success' && data.timestamp !== lastEsp32Timestamp) {
                lastEsp32Timestamp = data.timestamp;
                renderPredictionResults(data);
            }
        } catch (err) {
            // Ignore polling errors
        }
    }, 3000);

    // Live Microphone Recorder Setup
    const btnStartRecord = document.getElementById('btn-start-record');
    const btnStopRecord = document.getElementById('btn-stop-record');
    const micVisualizerOrb = document.getElementById('mic-visualizer-orb');
    const micStatusTitle = document.getElementById('mic-status-title');
    const micStatusDesc = document.getElementById('mic-status-desc');
    const recTimer = document.getElementById('rec-timer');

    let mediaRecorder = null;
    let audioChunks = [];
    let recInterval = null;
    let secondsLeft = 5;

    btnStartRecord.addEventListener('click', async () => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            mediaRecorder = new MediaRecorder(stream);
            audioChunks = [];

            mediaRecorder.ondataavailable = (e) => {
                if (e.data.size > 0) {
                    audioChunks.push(e.data);
                }
            };

            mediaRecorder.onstop = async () => {
                try {
                    micStatusTitle.textContent = 'Processing Recording...';
                    micStatusDesc.textContent = 'Encoding to 16kHz clinical PCM WAV...';

                    // Convert browser audio chunks to PCM AudioBuffer
                    const rawBlob = new Blob(audioChunks, { type: mediaRecorder.mimeType || 'audio/webm' });
                    const arrayBuffer = await rawBlob.arrayBuffer();
                    const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
                    const decodedBuffer = await audioCtx.decodeAudioData(arrayBuffer);

                    // Encode to authentic 16-bit 16kHz mono PCM WAV
                    const wavBlob = audioBufferToWavBlob(decodedBuffer, 16000);
                    const file = new File([wavBlob], `live_stethoscope_rec_${Date.now()}.wav`, { type: 'audio/wav' });

                    handleSelectedFile(file);
                    audioCtx.close();

                    // Reset UI
                    micVisualizerOrb.classList.remove('recording');
                    micStatusTitle.textContent = 'Live Recording Captured!';
                    micStatusDesc.textContent = 'Clinical 16kHz PCM audio ready for AI analysis';
                    btnStartRecord.classList.remove('hidden');
                    btnStopRecord.classList.add('hidden');
                    recTimer.classList.add('hidden');

                    // Auto Trigger Analysis
                    analyzeAudioFile();
                } catch (convErr) {
                    console.error('Audio conversion error:', convErr);
                    // Fallback to direct blob
                    const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                    const file = new File([audioBlob], `live_stethoscope_rec_${Date.now()}.wav`, { type: 'audio/wav' });
                    handleSelectedFile(file);
                    analyzeAudioFile();
                } finally {
                    // Stop mic track stream
                    stream.getTracks().forEach(track => track.stop());
                }
            };

            mediaRecorder.start();
            micVisualizerOrb.classList.add('recording');
            micStatusTitle.textContent = 'Recording Live Cardiac Audio...';
            micStatusDesc.textContent = 'Hold electronic stethoscope firmly against chest';
            btnStartRecord.classList.add('hidden');
            btnStopRecord.classList.remove('hidden');
            recTimer.classList.remove('hidden');

            secondsLeft = 5;
            recTimer.textContent = `00:0${secondsLeft}`;

            recInterval = setInterval(() => {
                secondsLeft--;
                recTimer.textContent = `00:0${secondsLeft}`;
                if (secondsLeft <= 0) {
                    clearInterval(recInterval);
                    if (mediaRecorder && mediaRecorder.state === 'recording') {
                        mediaRecorder.stop();
                    }
                }
            }, 1000);

        } catch (err) {
            console.error('Microphone Access Error:', err);
            alert('Microphone access denied or electronic stethoscope not detected: ' + err.message);
        }
    });

    btnStopRecord.addEventListener('click', () => {
        if (recInterval) clearInterval(recInterval);
        if (mediaRecorder && mediaRecorder.state === 'recording') {
            mediaRecorder.stop();
        }
    });

    let currentFile = null;

    // Dropzone Events
    dropzone.addEventListener('click', () => fileInput.click());

    dropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropzone.classList.add('dragover');
    });

    dropzone.addEventListener('dragleave', () => {
        dropzone.classList.remove('dragover');
    });

    dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropzone.classList.remove('dragover');
        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            handleSelectedFile(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files.length > 0) {
            handleSelectedFile(e.target.files[0]);
        }
    });

    btnClearFile.addEventListener('click', () => {
        clearSelectedFile();
    });

    function handleSelectedFile(file) {
        if (!file.name.toLowerCase().endsWith('.wav')) {
            alert('Please select a valid .wav heart audio file.');
            return;
        }
        currentFile = file;
        selectedFilename.textContent = file.name;
        filePreviewBar.classList.remove('hidden');
        dropzone.classList.add('hidden');

        // Load preview audio
        const fileURL = URL.createObjectURL(file);
        audioPlayer.src = fileURL;
        audioPlayerWrapper.classList.remove('hidden');

        btnAnalyze.disabled = false;
    }

    function clearSelectedFile() {
        currentFile = null;
        fileInput.value = '';
        audioPlayer.src = '';
        filePreviewBar.classList.add('hidden');
        audioPlayerWrapper.classList.add('hidden');
        dropzone.classList.remove('hidden');
        btnAnalyze.disabled = true;
    }

    // Sample Quick Selector Buttons
    document.querySelectorAll('.btn-sample').forEach(btn => {
        btn.addEventListener('click', async () => {
            const sampleType = btn.getAttribute('data-type');
            btn.style.opacity = '0.6';
            
            try {
                const response = await fetch('/api/samples');
                const samples = await response.json();
                const matched = samples.find(s => s.category.toLowerCase() === sampleType.toLowerCase());

                if (matched) {
                    const audioBlobResp = await fetch(matched.path);
                    const blob = await audioBlobResp.blob();
                    const file = new File([blob], matched.filename, { type: 'audio/wav' });
                    handleSelectedFile(file);
                    // Auto trigger analysis
                    analyzeAudioFile();
                } else {
                    alert('Sample file not found on server.');
                }
            } catch (err) {
                console.error('Error fetching sample:', err);
            } finally {
                btn.style.opacity = '1';
            }
        });
    });

    // Analyze Click Handler
    btnAnalyze.addEventListener('click', async () => {
        if (!currentFile) {
            // Auto-load demo sample if user clicks Analyze before picking a file
            try {
                const response = await fetch('/api/samples');
                const samples = await response.json();
                if (samples && samples.length > 0) {
                    const sample = samples[0];
                    const audioBlobResp = await fetch(sample.path);
                    const blob = await audioBlobResp.blob();
                    const file = new File([blob], sample.filename, { type: 'audio/wav' });
                    handleSelectedFile(file);
                }
            } catch (err) {
                console.error('Error auto-loading sample:', err);
            }
        }
        analyzeAudioFile();
    });

    async function analyzeAudioFile() {
        if (!currentFile) return;

        const toggleDenoise = document.getElementById('toggle-denoise');
        const isDenoise = toggleDenoise ? toggleDenoise.checked : true;

        // UI Loading state
        btnAnalyze.disabled = true;
        analyzeSpinner.classList.remove('hidden');

        const formData = new FormData();
        formData.append('file', currentFile);
        formData.append('denoise', isDenoise);

        try {
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 90000);

            const resp = await fetch('/api/predict', {
                method: 'POST',
                body: formData,
                signal: controller.signal
            });
            clearTimeout(timeoutId);

            const data = await resp.json();

            if (resp.ok && data.success) {
                displayResults(data);
            } else {
                alert('Analysis notice: ' + (data.error || 'Server status ' + resp.status));
            }
        } catch (err) {
            console.error('Prediction API Error:', err);
            if (err.name === 'AbortError') {
                alert('The analysis timed out. The cloud server may be initializing, please try again.');
            } else {
                alert('Connection notice: Cloud server is processing or waking up. Please retry in a few seconds.');
            }
        } finally {
            btnAnalyze.disabled = false;
            analyzeSpinner.classList.add('hidden');
        }
    }

    function audioBufferToWavBlob(audioBuffer, targetSampleRate = 16000) {
        const originalRate = audioBuffer.sampleRate;
        const channelData = audioBuffer.getChannelData(0);

        let outputData;
        if (originalRate === targetSampleRate) {
            outputData = channelData;
        } else {
            const ratio = originalRate / targetSampleRate;
            const newLength = Math.round(channelData.length / ratio);
            outputData = new Float32Array(newLength);
            for (let i = 0; i < newLength; i++) {
                const origIndex = Math.min(Math.floor(i * ratio), channelData.length - 1);
                outputData[i] = channelData[origIndex];
            }
        }

        const dataLength = outputData.length * 2;
        const buffer = new ArrayBuffer(44 + dataLength);
        const view = new DataView(buffer);

        function writeString(v, offset, str) {
            for (let i = 0; i < str.length; i++) {
                v.setUint8(offset + i, str.charCodeAt(i));
            }
        }

        // RIFF header
        writeString(view, 0, 'RIFF');
        view.setUint32(4, 36 + dataLength, true);
        writeString(view, 8, 'WAVE');

        // fmt subchunk
        writeString(view, 12, 'fmt ');
        view.setUint32(16, 16, true);
        view.setUint16(20, 1, true); // PCM
        view.setUint16(22, 1, true); // Mono
        view.setUint32(24, targetSampleRate, true);
        view.setUint32(28, targetSampleRate * 2, true);
        view.setUint16(32, 2, true);
        view.setUint16(34, 16, true);

        // data subchunk
        writeString(view, 36, 'data');
        view.setUint32(40, dataLength, true);

        let offset = 44;
        for (let i = 0; i < outputData.length; i++) {
            let s = Math.max(-1, Math.min(1, outputData[i]));
            view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
            offset += 2;
        }

        return new Blob([view], { type: 'audio/wav' });
    }

    function displayResults(data) {
        emptyState.classList.add('hidden');
        analysisResults.classList.remove('hidden');

        const info = data.info;
        predTitle.textContent = info.title;
        predDesc.textContent = info.description;
        predBadge.textContent = info.badge;
        predConf.textContent = `${data.confidence} Confidence`;

        // Style banner based on result
        predictionBanner.style.background = `${info.color}15`;
        predictionBanner.style.borderColor = `${info.color}40`;
        predBadge.style.background = `${info.color}25`;
        predBadge.style.color = info.color;

        // Set probabilities
        const probs = data.probabilities;
        valNormal.textContent = `${probs.normal.toFixed(1)}%`;
        barNormal.style.width = `${probs.normal}%`;

        valMurmur.textContent = `${probs.murmur_abnormal.toFixed(1)}%`;
        barMurmur.style.width = `${probs.murmur_abnormal}%`;

        valArtifact.textContent = `${probs.artifact.toFixed(1)}%`;
        barArtifact.style.width = `${probs.artifact}%`;

        // Visualizations
        imgWaveform.src = `data:image/png;base64,${data.waveform}`;
        imgSpectrogram.src = `data:image/png;base64,${data.spectrogram}`;
    }
});
