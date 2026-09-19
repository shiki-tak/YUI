import React from "react";
import ReactDOM from "react-dom/client";

// S0 で /broadcast・/control 画面をここから組み立てる。
// docs/design/yui_stream_first_design.md 7.「出力契約と配信用ブラウザ画面」を参照。
function App() {
  return <div>YUI</div>;
}

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
