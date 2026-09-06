"use client";

import { useState, useRef, useCallback, useEffect } from "react";
import { WebSocketClient, AnalysisResult } from "@/lib/websocket-client";
import RiskGauge from "@/components/RiskGauge";
import ConnectionStatus from "@/components/ConnectionStatus";
import ResultPanel from "@/components/ResultPanel";
import ExplanationPanel from "@/components/ExplanationPanel";

type ConnectionState = "disconnected" | "connecting" | "connected" | "processing" | "error";

export default function Home() {
  const [connectionState, setConnectionState] = useState<ConnectionState>("disconnected");
  const [latestResult, setLatestResult] = useState<AnalysisResult | null>(null);
  const [resultHistory, setResultHistory] = useState<AnalysisResult[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [isMicActive, setIsMicActive] = useState(false);
  const [isFileActive, setIsFileActive] = useState(false);
  const [fileName, setFileName] = useState<string | null>(null);
  const [fileProgress, setFileProgress] = useState<string | null>(null);
  const [waveformData, setWaveformData] = useState<number[]>(new Array(32).fill(0));

  const wsClientRef = useRef<WebSocketClient | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const animationFrameRef = useRef<number | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const stopStreamingRef = useRef(false);

  const createClient = useCallback(
    (url: string): WebSocketClient => {
      const client = new WebSocketClient(
        url,
        (result) => {
          setLatestResult(result);
          setResultHistory((prev) => [...prev.slice(-49), result]);
          setConnectionState("processing");
        },
        (err) => {
          setError(err);
          setConnectionState("error");
          stopAnalysis();
        },
        () => {
          setConnectionState("disconnected");
        }
      );
      wsClientRef.current = client;
      return client;
    },
    []
  );

  const decodeAudioFile = useCallback(async (file: File): Promise<Int16Array> => {
    const arrayBuffer = await file.arrayBuffer();
    const ac = new AudioContext({ sampleRate: 16000 });
    let buffer = await ac.decodeAudioData(arrayBuffer);
    if (buffer.sampleRate !== 16000) {
      const targetLen = Math.ceil(buffer.duration * 16000);
      const offline = new OfflineAudioContext(1, targetLen, 16000);
      const source = offline.createBufferSource();
      source.buffer = buffer;
      source.connect(offline.destination);
      source.start(0);
      buffer = await offline.startRendering();
    }
    void ac.close();

    const channels = Math.min(buffer.numberOfChannels, 2);
    const int16 = new Int16Array(buffer.length);
    for (let i = 0; i < buffer.length; i++) {
      let s = 0;
      for (let c = 0; c < channels; c++) s += buffer.getChannelData(c)[i];
      s /= channels;
      s = Math.max(-1, Math.min(1, s));
      int16[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
    }
    return int16;
  }, []);

  const processAudioFrame = useCallback(
    (audioBuffer: AudioBuffer) => {
      if (!wsClientRef.current?.isConnected) return;

      const channelData = audioBuffer.getChannelData(0);
      const int16 = new Int16Array(channelData.length);
      for (let i = 0; i < channelData.length; i++) {
        const s = Math.max(-1, Math.min(1, channelData[i]));
        int16[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
      }
      wsClientRef.current.sendAudio(int16);
    },
    []
  );

  const stopAnalysis = useCallback(() => {
    stopStreamingRef.current = true;
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
      animationFrameRef.current = null;
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    if (audioContextRef.current) {
      audioContextRef.current.close();
      audioContextRef.current = null;
    }
    if (wsClientRef.current) {
      wsClientRef.current.disconnect();
      wsClientRef.current = null;
    }
    setIsMicActive(false);
    setIsFileActive(false);
    setFileProgress(null);
    setFileName(null);
    setConnectionState("disconnected");
    setWaveformData(new Array(32).fill(0));
  }, []);

  const startAnalysis = useCallback(async () => {
    try {
      setError(null);
      setLatestResult(null);
      setResultHistory([]);
      setConnectionState("connecting");
      stopStreamingRef.current = false;

      const wsUrl =
        process.env.NEXT_PUBLIC_WS_URL ||
        "ws://localhost:8000/ws/analyze";

      const client = createClient(wsUrl);
      await client.connect();

      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          sampleRate: 16000,
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: false,
        },
      });
      streamRef.current = stream;

      const audioContext = new AudioContext({ sampleRate: 16000 });
      audioContextRef.current = audioContext;
      const source = audioContext.createMediaStreamSource(stream);

      const analyser = audioContext.createAnalyser();
      analyser.fftSize = 64;
      source.connect(analyser);
      analyserRef.current = analyser;

      setConnectionState("connected");
      setIsMicActive(true);

      const bufferSize = 4096;
      const scriptNode = audioContext.createScriptProcessor(bufferSize, 1, 1);
      source.connect(scriptNode);
      scriptNode.connect(audioContext.destination);

      scriptNode.onaudioprocess = (event) => {
        if (!wsClientRef.current?.isConnected) return;
        const inputBuffer = event.inputBuffer;
        processAudioFrame(inputBuffer);
      };

      const updateWaveform = () => {
        if (!analyserRef.current) return;
        const dataArray = new Uint8Array(analyser.frequencyBinCount);
        analyser.getByteFrequencyData(dataArray);
        const normalized = Array.from(dataArray).map((v) => v / 255);
        setWaveformData(normalized);
        animationFrameRef.current = requestAnimationFrame(updateWaveform);
      };
      updateWaveform();
    } catch (err) {
      const message = err instanceof Error ? err.message : "Failed to start analysis";
      setError(message);
      setConnectionState("error");
    }
  }, [createClient, processAudioFrame]);

  const startFileAnalysis = useCallback(
    async (file: File) => {
      try {
        setError(null);
        setLatestResult(null);
        setResultHistory([]);
        setConnectionState("connecting");
        stopStreamingRef.current = false;
        setFileName(file.name);

        const wsUrl =
          process.env.NEXT_PUBLIC_WS_URL || "ws://localhost:8000/ws/analyze";
        const client = createClient(wsUrl);
        await client.connect();
        setConnectionState("connected");
        setIsFileActive(true);

        const pcm = await decodeAudioFile(file);
        const bytesPerFrame = (16000 / 4) * 2; // 250ms = 4000 samples = 8000 bytes
        const chunks = Math.ceil(pcm.byteLength / bytesPerFrame);

        for (let i = 0; i < chunks; i++) {
          if (!wsClientRef.current?.isConnected || stopStreamingRef.current) break;
          const start = i * bytesPerFrame;
          const chunk = pcm.subarray(
            Math.floor(start / 2),
            Math.floor(Math.min(pcm.byteLength, start + bytesPerFrame) / 2)
          );
          wsClientRef.current.sendAudio(chunk);
          setFileProgress(`Chunk ${i + 1}/${chunks}`);
          await new Promise((r) => setTimeout(r, 240));
        }

        setFileProgress(null);
        await new Promise((r) => setTimeout(r, 1500));
        if (!stopStreamingRef.current) stopAnalysis();
      } catch (err) {
        const message = err instanceof Error ? err.message : "Failed to analyze file";
        setError(message);
        setConnectionState("error");
        stopAnalysis();
      }
    },
    [createClient, decodeAudioFile, stopAnalysis]
  );

  const handleFileChange = useCallback(
    (event: React.ChangeEvent<HTMLInputElement>) => {
      const file = event.target.files?.[0];
      event.target.value = "";
      if (file && !isMicActive && !isFileActive) {
        void startFileAnalysis(file);
      }
    },
    [startFileAnalysis, isMicActive, isFileActive]
  );

  useEffect(() => {
    return () => stopAnalysis();
  }, [stopAnalysis]);

  return (
    <main className="min-h-screen flex flex-col items-center px-4 py-8">
      <header className="text-center mb-8">
        <h1 className="text-3xl font-bold tracking-tight text-white mb-2">
          Voice Clone Detector
        </h1>
        <p className="text-slate-400 text-sm max-w-lg">
          Real-time voice authenticity analysis. Your audio is streamed securely to a
          GPU server and never stored.
        </p>
      </header>

      <div className="flex items-center gap-4 mb-4">
        <ConnectionStatus state={connectionState} />
        <button
          onClick={isMicActive || isFileActive ? stopAnalysis : startAnalysis}
          disabled={isFileActive}
          className={`px-6 py-3 rounded-xl font-semibold text-sm transition-all disabled:opacity-40 ${
            isMicActive || isFileActive
              ? "bg-red-600 hover:bg-red-700 text-white"
              : "bg-blue-600 hover:bg-blue-700 text-white"
          } ${(connectionState === "connected" && isMicActive) || isFileActive ? "pulse-active" : ""}`}
        >
          {isMicActive || isFileActive ? "Stop Analysis" : "Start Analysis"}
        </button>

        <label
          className={`px-6 py-3 rounded-xl font-semibold text-sm transition-all cursor-pointer ${
            isMicActive || isFileActive
              ? "bg-slate-700 text-slate-400 pointer-events-none"
              : "bg-emerald-600 hover:bg-emerald-700 text-white"
          }`}
        >
          {isFileActive ? "Analyzing File..." : "Analyze Audio File"}
          <input
            type="file"
            accept="audio/*,.mp3,.wav,.m4a,.ogg,.flac,.aac,.webm"
            className="hidden"
            onChange={handleFileChange}
            disabled={isMicActive || isFileActive}
          />
        </label>
      </div>

      {(fileName || fileProgress) && (
        <p className="text-xs text-emerald-300 mb-6 max-w-md truncate">
          {fileName}
          {fileProgress ? ` — ${fileProgress}` : ""}
        </p>
      )}

      {error && (
        <div className="glass-card rounded-xl px-4 py-3 mb-6 text-red-400 text-sm max-w-md">
          {error}
        </div>
      )}

      <div className="flex flex-col lg:flex-row gap-6 w-full max-w-5xl">
        <div className="flex flex-col items-center gap-6 lg:w-1/3">
          <RiskGauge
            riskScore={latestResult?.risk_score ?? 0}
            verdict={latestResult?.verdict ?? null}
          />

          <div className="glass-card rounded-xl p-4 w-full">
            <p className="text-xs text-slate-400 mb-2 text-center uppercase tracking-wider">
              Audio Activity
            </p>
            <div className="flex items-end justify-center gap-[3px] h-12">
              {waveformData.slice(0, 32).map((v, i) => (
                <div
                  key={i}
                  className="w-1.5 bg-blue-500 rounded-t-sm"
                  style={{
                    height: `${Math.max(4, v * 48)}px`,
                    opacity: isMicActive ? 1 : 0.3,
                    transition: "height 0.1s ease, opacity 0.3s ease",
                  }}
                />
              ))}
            </div>
          </div>
        </div>

        <div className="flex-1 flex flex-col gap-6">
          {latestResult && <ResultPanel result={latestResult} />}
          {latestResult && <ExplanationPanel result={latestResult} />}
        </div>
      </div>

      {resultHistory.length > 1 && (
        <div className="glass-card rounded-xl p-4 w-full max-w-5xl mt-6">
          <p className="text-xs text-slate-400 mb-3 uppercase tracking-wider">
            Recent Windows ({resultHistory.length})
          </p>
          <div className="flex gap-1 items-end h-16">
            {resultHistory.slice(-60).map((r, i) => (
              <div
                key={i}
                className="flex-1 rounded-t-sm"
                style={{
                  height: `${Math.max(4, r.risk_score * 0.6)}px`,
                  backgroundColor:
                    r.verdict === "FAKE"
                      ? "#ef4444"
                      : r.verdict === "REAL"
                      ? "#22c55e"
                      : "#eab308",
                  opacity: 0.8,
                }}
              />
            ))}
          </div>
        </div>
      )}

      <footer className="mt-12 text-center text-xs text-slate-600">
        <p>Advisory signal only — not a definitive authenticity verdict.</p>
        <p className="mt-1">SIH26104 Voice Cloning Detection System v0.3</p>
      </footer>
    </main>
  );
}
