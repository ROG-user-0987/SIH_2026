export interface Signal {
  name: string;
  category: string;
  score: number;
  weight: number;
  top_feature?: string;
}

export interface Explanation {
  dominant_signal: string;
  dominant_category: string;
  rationale_tags: string[];
  explainability_version: string;
}

export interface AnalysisResult {
  type: string;
  session_id?: string;
  window_ts_start: string;
  window_ts_end: string;
  risk_score: number;
  verdict: "REAL" | "FAKE" | "INSUFFICIENT_SPEECH" | "UNAVAILABLE";
  confidence: number;
  signals: Signal[];
  optional_signals: Record<string, unknown>;
  explanation: Explanation;
  model_versions: Record<string, string>;
}

type ResultCallback = (result: AnalysisResult) => void;
type ErrorCallback = (error: string) => void;
type CloseCallback = () => void;

export class WebSocketClient {
  private ws: WebSocket | null = null;
  private sessionId: string;
  private onResult: ResultCallback;
  private onError: ErrorCallback;
  private onClose: CloseCallback;
  private _isConnected = false;

  constructor(
    url: string,
    onResult: ResultCallback,
    onError: ErrorCallback,
    onClose: CloseCallback
  ) {
    this.onResult = onResult;
    this.onError = onError;
    this.onClose = onClose;
    this.sessionId = crypto.randomUUID().slice(0, 8);
    void url;
  }

  get isConnected(): boolean {
    return this._isConnected && this.ws?.readyState === WebSocket.OPEN;
  }

  async connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      const wsUrl =
        process.env.NEXT_PUBLIC_WS_URL || "ws://localhost:8000/ws/analyze";
      this.ws = new WebSocket(wsUrl);

      this.ws.binaryType = "arraybuffer";

      this.ws.onopen = () => {
        const initPayload = JSON.stringify({
          type: "session_init",
          session_id: this.sessionId,
          sample_rate: 16000,
        });
        this.ws!.send(initPayload);

        const checkReady = setInterval(() => {
          if (this._isConnected) {
            clearInterval(checkReady);
            resolve();
          }
        }, 50);

        setTimeout(() => {
          clearInterval(checkReady);
          if (!this._isConnected) {
            reject(new Error("Session init timeout"));
          }
        }, 5000);
      };

      this.ws.onmessage = (event: MessageEvent) => {
        if (typeof event.data === "string") {
          try {
            const msg = JSON.parse(event.data);
            if (msg.type === "session_ack") {
              this._isConnected = true;
            } else if (msg.type === "result") {
              this.onResult(msg as AnalysisResult);
            } else if (msg.type === "error") {
              this.onError(msg.message || "Server error");
            }
          } catch {
            // binary frame echo or malformed — ignore
          }
        }
      };

      this.ws.onerror = () => {
        this.onError("WebSocket connection error");
        this._isConnected = false;
        reject(new Error("WebSocket connection error"));
      };

      this.ws.onclose = () => {
        this._isConnected = false;
        this.onClose();
      };
    });
  }

  sendAudio(buffer: ArrayBuffer | ArrayBufferView): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(buffer);
    }
  }

  disconnect(): void {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
    this._isConnected = false;
  }
}
