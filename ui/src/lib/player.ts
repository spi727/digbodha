import { useCallback, useEffect, useRef, useState } from "react";
import type { RunMeta } from "./data";

// Frame-accurate playback over either a video file or a folder of frames.
// The pipeline's outputs are indexed by frame, so everything downstream keys off `frame`.

export interface Player {
  frame: number;
  playing: boolean;
  rate: number;
  ready: boolean;
  setRate: (r: number) => void;
  play: () => void;
  pause: () => void;
  toggle: () => void;
  seek: (f: number) => void;
  step: (d: number) => void;
  videoRef: (el: HTMLVideoElement | null) => void;
}

type VideoWithRVFC = HTMLVideoElement & {
  requestVideoFrameCallback?: (cb: (now: number, md: { mediaTime: number }) => void) => number;
  cancelVideoFrameCallback?: (h: number) => void;
};

export function usePlayer(run: RunMeta | null, opts: { loop?: boolean; autoplay?: boolean } = {}): Player {
  const [frame, setFrame] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [rate, setRateState] = useState(1);
  const [ready, setReady] = useState(false);
  const [video, setVideo] = useState<VideoWithRVFC | null>(null);
  const frameRef = useRef(0);
  frameRef.current = frame;

  const fps = run?.fps ?? 10;
  const n = run?.frames ?? 1;
  const kind = run?.source.kind ?? "none";
  const clamp = useCallback((f: number) => Math.max(0, Math.min(n - 1, Math.round(f))), [n]);
  const toFrame = useCallback((t: number) => clamp(Math.floor(t * fps + 1e-3)), [clamp, fps]);

  // reset on run change
  useEffect(() => {
    setFrame(0);
    setPlaying(false);
    setReady(kind !== "video");
  }, [run?.name, kind]);

  // ---------------------------------------------------------- video source
  useEffect(() => {
    if (kind !== "video" || !video) return;
    let handle = 0, raf = 0, alive = true;
    const tick = (_: number, md: { mediaTime: number }) => {
      if (!alive) return;
      setFrame(toFrame(md.mediaTime));
      handle = video.requestVideoFrameCallback!(tick);
    };
    const rafTick = () => {
      if (!alive) return;
      setFrame(toFrame(video.currentTime));
      raf = requestAnimationFrame(rafTick);
    };
    if (video.requestVideoFrameCallback) handle = video.requestVideoFrameCallback(tick);
    else raf = requestAnimationFrame(rafTick);

    const onPlay = () => setPlaying(true);
    const onPause = () => setPlaying(false);
    const onSeeked = () => setFrame(toFrame(video.currentTime));
    const onReady = () => setReady(true);
    video.addEventListener("play", onPlay);
    video.addEventListener("pause", onPause);
    video.addEventListener("ended", onPause);
    video.addEventListener("seeked", onSeeked);
    video.addEventListener("loadeddata", onReady);
    if (video.readyState >= 2) setReady(true);
    if (opts.autoplay) video.play().catch(() => setPlaying(false));
    return () => {
      alive = false;
      if (handle && video.cancelVideoFrameCallback) video.cancelVideoFrameCallback(handle);
      cancelAnimationFrame(raf);
      video.removeEventListener("play", onPlay);
      video.removeEventListener("pause", onPause);
      video.removeEventListener("ended", onPause);
      video.removeEventListener("seeked", onSeeked);
      video.removeEventListener("loadeddata", onReady);
    };
  }, [video, kind, toFrame, opts.autoplay]);

  useEffect(() => {
    if (video) video.playbackRate = rate;
  }, [video, rate]);

  // ---------------------------------------------------------- frame-folder source
  useEffect(() => {
    if (kind !== "frames" || !playing) return;
    const id = window.setInterval(() => {
      const next = frameRef.current + 1;
      if (next >= n) {
        if (opts.loop) setFrame(0);
        else setPlaying(false);
      } else setFrame(next);
    }, 1000 / (fps * rate));
    return () => window.clearInterval(id);
  }, [kind, playing, fps, rate, n, opts.loop]);

  useEffect(() => {
    if (kind === "frames" && opts.autoplay) setPlaying(true);
  }, [kind, opts.autoplay]);

  // ---------------------------------------------------------- controls
  const seek = useCallback((f: number) => {
    const g = clamp(f);
    setFrame(g);
    if (kind === "video" && video) video.currentTime = (g + 0.5) / fps;
  }, [clamp, kind, video, fps]);

  const play = useCallback(() => {
    if (kind === "video" && video) {
      if (frameRef.current >= n - 1) video.currentTime = 0;
      video.play().catch(() => {});
    } else if (kind === "frames") {
      if (frameRef.current >= n - 1) setFrame(0);
      setPlaying(true);
    }
  }, [kind, video, n]);

  const pause = useCallback(() => {
    if (kind === "video" && video) video.pause();
    else setPlaying(false);
  }, [kind, video]);

  const toggle = useCallback(() => (playing ? pause() : play()), [playing, play, pause]);
  const step = useCallback((d: number) => {
    pause();
    seek(frameRef.current + d);
  }, [pause, seek]);

  const videoRef = useCallback((el: HTMLVideoElement | null) => setVideo(el as VideoWithRVFC | null), []);

  return { frame, playing, rate, ready, setRate: setRateState, play, pause, toggle, seek, step, videoRef };
}
