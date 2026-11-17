import { useEffect, useState } from "react";
import { Landing } from "./pages/Landing";
import { Viewer } from "./pages/Viewer";
import { Evaluation } from "./pages/Evaluation";
import { Analyze } from "./pages/Analyze";
import { Icon } from "./components/Icon";
import { DEMO_NAME } from "./lib/data";

function useRoute() {
  const read = () => location.hash.replace(/^#\/?/, "").split("/").filter(Boolean);
  const [route, setRoute] = useState(read);
  useEffect(() => {
    const on = () => {
      setRoute(read());
      window.scrollTo({ top: 0 });
    };
    window.addEventListener("hashchange", on);
    return () => window.removeEventListener("hashchange", on);
  }, []);
  return route;
}

export function App() {
  const route = useRoute();
  const page = route[0] ?? "";
  let body: JSX.Element;
  if (page === "run") body = <Viewer name={decodeURIComponent(route[1] ?? DEMO_NAME)} />;
  else if (page === "evaluation") body = <Evaluation />;
  else if (page === "analyze") body = <Analyze />;
  else body = <Landing />;

  useEffect(() => {
    const titles: Record<string, string> = { run: "Run viewer", evaluation: "Evaluation", analyze: "Analyze" };
    document.title = titles[page] ? `${titles[page]} · DigBodha` : "DigBodha · See what’s coming";
  }, [page]);

  return (
    <div className="app">
      <nav className="nav">
        <a href="#/" className="brand" aria-label="DigBodha home">
          <Mark />
          <span className="brand__name">DigBodha</span>
          <span className="brand__ver mono">1</span>
        </a>
        <div className="nav__links">
          <a href={`#/run/${DEMO_NAME}`} className={page === "run" ? "is-active" : ""}>Viewer</a>
          <a href="#/evaluation" className={page === "evaluation" ? "is-active" : ""}>Evaluation</a>
          <a href="#/analyze" className={page === "analyze" ? "is-active" : ""}>Analyze</a>
          <a href="https://github.com/spi727/digbodha" target="_blank" rel="noreferrer" className="nav__gh" aria-label="Source on GitHub">
            <Icon name="github" size={18} />
          </a>
        </div>
      </nav>
      <main>{body}</main>
      <footer className="footer">
        <div className="footer__brand">
          <Mark />
          <div>
            <b>DigBodha 1</b>
            <p className="muted small">Text-prompted motion and collision-risk estimation from a single camera.</p>
          </div>
        </div>
        <div className="footer__cols small">
          <div>
            <span className="muted">Team</span>
            <span>K. Praveen Sagar</span>
            <span>Arnav Abhijit Rajurkar</span>
            <span>Madhur Vijay Patil</span>
          </div>
          <div>
            <span className="muted">Context</span>
            <span>Computer Vision course project</span>
            <span>M.Tech CSE (Data Science &amp; AI)</span>
            <span>IIITDM Kancheepuram · 2026</span>
          </div>
          <div>
            <span className="muted">Built on</span>
            <span>SAM 3.1 (Meta)</span>
            <span>KITTI Tracking (CC BY-NC-SA 3.0)</span>
            <a href="https://github.com/spi727/digbodha" target="_blank" rel="noreferrer">Source on GitHub ↗</a>
          </div>
        </div>
      </footer>
    </div>
  );
}

// A compass needle inside an aperture: dig (direction) + bodha (awareness).
function Mark() {
  return (
    <svg width="22" height="22" viewBox="0 0 24 24" className="mark" aria-hidden="true">
      <circle cx="12" cy="12" r="10.25" fill="none" stroke="currentColor" strokeOpacity="0.35" strokeWidth="1.5" />
      <path d="M12 3.5 15 12h-6z" fill="var(--danger)" />
      <path d="M12 20.5 9 12h6z" fill="currentColor" />
      <circle cx="12" cy="12" r="1.6" fill="var(--bg)" />
    </svg>
  );
}
