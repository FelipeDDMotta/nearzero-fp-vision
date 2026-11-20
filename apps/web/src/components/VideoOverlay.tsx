"use client";

import { forwardRef, useEffect, useImperativeHandle, useRef, useState } from "react";
import { displayStatus, STATUS_STYLE, type DisplayStatus, type Track } from "@/lib/types";

export interface VideoHandle {
  seek: (t: number) => void;
}

interface Props {
  src: string;
  tracks: Track[];
  visible: Set<DisplayStatus>;
  selectedId: number | null;
  onSelect: (id: number) => void;
}

/** Linear interpolation of a track's box at time t (boxes are [t, x1, y1, x2, y2, conf], normalised). */
function boxAt(track: Track, t: number): [number, number, number, number, number] | null {
  const b = track.boxes;
  if (b.length === 1) {
    // single-frame candidates: hold the box on screen for a moment so it is visible
    return t >= b[0][0] - 0.1 && t <= b[0][0] + 1.0 ? [b[0][1], b[0][2], b[0][3], b[0][4], b[0][5]] : null;
  }
  if (!b.length || t < b[0][0] - 0.05 || t > b[b.length - 1][0] + 0.25) return null;
  let lo = 0,
    hi = b.length - 1;
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    if (b[mid][0] <= t) lo = mid;
    else hi = mid - 1;
  }
  const a = b[lo];
  const n = b[lo + 1];
  if (!n || n[0] - a[0] > 1.0 || t <= a[0]) return [a[1], a[2], a[3], a[4], a[5]];
  const k = (t - a[0]) / (n[0] - a[0]);
  return [a[1] + (n[1] - a[1]) * k, a[2] + (n[2] - a[2]) * k, a[3] + (n[3] - a[3]) * k, a[4] + (n[4] - a[4]) * k, a[5]];
}

export const VideoOverlay = forwardRef<VideoHandle, Props>(function VideoOverlay({ src, tracks, visible, selectedId, onSelect }, ref) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const state = useRef({ tracks, visible, selectedId });
  state.current = { tracks, visible, selectedId };
  const hitboxes = useRef<{ id: number; x: number; y: number; w: number; h: number }[]>([]);

  useImperativeHandle(ref, () => ({
    seek: (t: number) => {
      const v = videoRef.current;
      if (v) {
        v.currentTime = Math.max(0, t);
        v.pause();
      }
    },
  }));

  useEffect(() => {
    const video = videoRef.current!;
    const canvas = canvasRef.current!;
    const ctx = canvas.getContext("2d")!;
    let raf = 0;
    let stopped = false;

    const draw = () => {
      const { tracks, visible, selectedId } = state.current;
      const rect = canvas.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      if (canvas.width !== Math.round(rect.width * dpr) || canvas.height !== Math.round(rect.height * dpr)) {
        canvas.width = Math.round(rect.width * dpr);
        canvas.height = Math.round(rect.height * dpr);
      }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, rect.width, rect.height);
      // object-fit: contain -> compute the drawn video rectangle
      const vw = video.videoWidth || 16,
        vh = video.videoHeight || 9;
      const s = Math.min(rect.width / vw, rect.height / vh);
      const dw = vw * s,
        dh = vh * s,
        ox = (rect.width - dw) / 2,
        oy = (rect.height - dh) / 2;
      const t = video.currentTime;
      const hits: typeof hitboxes.current = [];
      ctx.font = "600 11px ui-monospace, SFMono-Regular, Menlo, monospace";
      for (const tr of tracks) {
        const st = displayStatus(tr);
        if (!visible.has(st) && tr.id !== selectedId) continue;
        const bx = boxAt(tr, t);
        if (!bx) continue;
        const [x1, y1, x2, y2, conf] = bx;
        const x = ox + x1 * dw,
          y = oy + y1 * dh,
          w = (x2 - x1) * dw,
          h = (y2 - y1) * dh;
        const style = STATUS_STYLE[st];
        const sel = tr.id === selectedId;
        ctx.save();
        ctx.lineWidth = sel ? 3 : 1.75;
        ctx.strokeStyle = style.color;
        ctx.setLineDash(style.dash || []);
        if (sel) {
          ctx.shadowColor = style.color;
          ctx.shadowBlur = 12;
        }
        ctx.strokeRect(x, y, w, h);
        ctx.restore();
        const corrected = tr.triage?.final.corrected_class;
        const cls = corrected && corrected !== "none" ? corrected : tr.cls;
        const label = `${cls} ${conf.toFixed(2)}${st === "rejected" ? " FP" : st === "escalated" ? " ?" : ""}`;
        const tw = ctx.measureText(label).width + 8;
        ctx.fillStyle = style.color;
        ctx.globalAlpha = sel ? 1 : 0.85;
        ctx.fillRect(x, Math.max(0, y - 16), tw, 16);
        ctx.globalAlpha = 1;
        ctx.fillStyle = "#05080c";
        ctx.fillText(label, x + 4, Math.max(12, y - 4));
        hits.push({ id: tr.id, x, y, w, h });
      }
      hitboxes.current = hits;
    };

    const loop = () => {
      if (stopped) return;
      draw();
      raf = requestAnimationFrame(loop);
    };
    loop();
    return () => {
      stopped = true;
      cancelAnimationFrame(raf);
    };
  }, []);

  const onClick = (e: React.MouseEvent) => {
    const r = canvasRef.current!.getBoundingClientRect();
    const x = e.clientX - r.left,
      y = e.clientY - r.top;
    const hit = [...hitboxes.current].reverse().find((h) => x >= h.x && x <= h.x + h.w && y >= h.y && y <= h.y + h.h);
    const v = videoRef.current;
    if (hit) {
      onSelect(hit.id);
      v?.pause();
    } else if (v) {
      if (v.paused) v.play();
      else v.pause();
    }
  };

  return (
    <div className="overflow-hidden rounded-lg border border-line bg-black">
      <div className="relative aspect-video w-full">
        <video ref={videoRef} src={src} className="absolute inset-0 h-full w-full object-contain" muted playsInline loop autoPlay />
        <canvas ref={canvasRef} onClick={onClick} className="absolute inset-0 h-full w-full cursor-crosshair" />
      </div>
      <Controls videoRef={videoRef} tracks={tracks} visible={visible} onSelect={onSelect} />
    </div>
  );
});

function fmt(t: number) {
  const m = Math.floor(t / 60);
  const s = Math.floor(t % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

function Controls({
  videoRef,
  tracks,
  visible,
  onSelect,
}: {
  videoRef: React.RefObject<HTMLVideoElement | null>;
  tracks: Track[];
  visible: Set<DisplayStatus>;
  onSelect: (id: number) => void;
}) {
  const [t, setT] = useState(0);
  const [dur, setDur] = useState(0);
  const [playing, setPlaying] = useState(true);
  useEffect(() => {
    const v = videoRef.current;
    if (!v) return;
    const upd = () => {
      setT(v.currentTime);
      setDur(v.duration || 0);
      setPlaying(!v.paused);
    };
    const id = setInterval(upd, 200);
    v.addEventListener("play", upd);
    v.addEventListener("pause", upd);
    return () => {
      clearInterval(id);
      v.removeEventListener("play", upd);
      v.removeEventListener("pause", upd);
    };
  }, [videoRef]);
  const toggle = () => {
    const v = videoRef.current;
    if (!v) return;
    if (v.paused) v.play();
    else v.pause();
  };
  const markers = tracks.filter((tr) => {
    const st = displayStatus(tr);
    return st !== "auto" && st !== "dropped" && visible.has(st);
  });
  return (
    <div className="flex items-center gap-3 border-t border-line bg-panel px-3 py-2 text-xs">
      <button onClick={toggle} className="w-12 rounded bg-panel-2 px-2 py-1 font-mono hover:bg-line" aria-label={playing ? "Pause" : "Play"}>
        {playing ? "||" : "▶"}
      </button>
      <span className="w-20 font-mono text-muted">
        {fmt(t)} / {fmt(dur)}
      </span>
      <div className="relative h-6 flex-1">
        <div className="absolute inset-x-0 top-1/2 h-1 -translate-y-1/2 rounded bg-line" />
        <div className="absolute left-0 top-1/2 h-1 -translate-y-1/2 rounded bg-accent/60" style={{ width: `${dur ? (100 * t) / dur : 0}%` }} />
        <input
          type="range"
          min={0}
          max={dur || 1}
          step={0.05}
          value={t}
          onChange={(e) => {
            const v = videoRef.current;
            if (v) v.currentTime = Number(e.target.value);
          }}
          className="absolute inset-x-0 top-0 h-6 w-full cursor-pointer opacity-0"
        />
        {dur > 0 &&
          markers.map((m) => (
            <button
              key={m.id}
              title={`#${m.id} ${m.cls} @ ${m.t0.toFixed(1)}s`}
              onClick={() => {
                const v = videoRef.current;
                if (v) {
                  v.currentTime = m.t0;
                  v.pause();
                }
                onSelect(m.id);
              }}
              className="absolute top-0 z-10 h-6 w-[3px] -translate-x-1/2 rounded-sm opacity-80 hover:opacity-100"
              style={{ left: `${(100 * m.t0) / dur}%`, background: STATUS_STYLE[displayStatus(m)].color }}
            />
          ))}
      </div>
    </div>
  );
}
