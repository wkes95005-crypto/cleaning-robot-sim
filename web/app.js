"use strict";

const CS = 18; // canvas pixels per cell
const ROLE_COLOR = { sweeper: "#3ccf8e", mopper: "#4f8dff", arm: "#f2c94c" };
const DIRT_COLOR = { dry: "#c79a5b", wet: "#39b6ff", mixed: "#b07cff" };
const OBS_COLOR = { box: "#d9822b", cart: "#9aa7b8", pallet: "#8a6a43", cable: "#f2c94c" };
const ZONE_TINT = ["#2b3a52", "#2f3d3a", "#3a3346", "#26303d", "#33402f", "#2c3540", "#3d3a2c",
                   "#2f3a4a", "#40332f", "#2e3f46", "#3a2e3d"];
const STATE_LABEL = { idle: "待命", moving: "移動中", waiting: "讓行等待", cleaning: "清潔中",
                      grasping: "夾取中", placing: "放置中", charging: "充電中" };
const KIND_LABEL = { sweep: "掃地", mop: "拖地", patrol: "巡邏", clear: "搬移" };
const LOG_CATS = { voice: "語音", llm: "LLM", dispatch: "派工", vision: "辨識", obstacle: "障礙",
                   energy: "電量", conflict: "衝突", clean: "清潔" };
const PRESETS = {
  sweeper: ["去掃貨架A區", "這裡有人打翻飲料，處理一下", "先清理生鮮區，避開地上的電線，最後再掃收銀區"],
  mopper: ["入口地板濕了，拖一下", "去拖收銀區", "促銷區和入口同時拖地"],
  arm: ["貨架B區有紙箱擋路，請搬開", "去巡查促銷區有沒有障礙物"],
  common: ["停下來", "繼續工作", "回充電站", "全部回充電站待命", "大家繼續工作"],
};

const canvas = document.getElementById("map");
const ctx = canvas.getContext("2d");
let MAP = null;
let STATE = null;
let logFilter = new Set(Object.keys(LOG_CATS).filter((c) => c !== "clean"));
let lastLogKey = "";
let selected = null;   // robot currently being talked to
let lastChatKey = "";

async function api(path, body) {
  const opt = body ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) } : {};
  const r = await fetch(path, opt);
  return r.json();
}

// ------------------------------------------------------------------ setup
async function init() {
  MAP = await api("/api/map");
  canvas.width = MAP.w * CS;
  canvas.height = MAP.h * CS;

  const tabs = document.getElementById("talk-tabs");
  MAP.fleet.forEach((r, i) => {
    const b = document.createElement("button");
    b.className = "talk-tab";
    b.dataset.id = r.id;
    b.style.setProperty("--tab-color", ROLE_COLOR[r.role]);
    b.title = `數字鍵 ${i + 1}`;
    b.innerHTML = `<span class="top"><i class="dot" style="background:${ROLE_COLOR[r.role]}"></i>${r.id}</span>
      <span class="sub">${r.label}</span>`;
    b.onclick = () => selectRobot(r.id);
    tabs.append(b);
  });
  document.addEventListener("keydown", (e) => {
    const typing = document.activeElement && document.activeElement.id === "cmd-text";
    const n = Number(e.key);
    if ((!typing || e.altKey) && n >= 1 && n <= MAP.fleet.length) {
      e.preventDefault();
      selectRobot(MAP.fleet[n - 1].id, false); // keep focus where it is so the next digit switches again
    }
  });
  setupMic();
  setupCameras();
  selectRobot(MAP.fleet.find((r) => r.role === "sweeper").id, false);

  const filter = document.getElementById("log-filter");
  for (const [cat, label] of Object.entries(LOG_CATS)) {
    const l = document.createElement("label");
    l.innerHTML = `<input type="checkbox" ${logFilter.has(cat) ? "checked" : ""}> ${label}`;
    l.querySelector("input").onchange = (e) => {
      e.target.checked ? logFilter.add(cat) : logFilter.delete(cat);
      lastLogKey = "";
      renderLog();
    };
    filter.append(l);
  }

  document.getElementById("cmd-send").onclick = sendCommand;
  document.getElementById("cmd-text").addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.isComposing) sendCommand(); });
  document.getElementById("btn-pause").onclick = () => api("/api/control", { action: STATE && STATE.paused ? "resume" : "pause" });
  document.getElementById("btn-step").onclick = () => api("/api/control", { action: "step" });
  document.getElementById("btn-reset").onclick = () => { lastLogKey = ""; api("/api/control", { action: "reset" }); };
  const speed = document.getElementById("speed");
  speed.oninput = () => {
    document.getElementById("speed-val").textContent = speed.value;
    api("/api/control", { action: "speed", value: Number(speed.value) });
  };
  canvas.addEventListener("click", onMapClick);
  poll();
}

async function sendCommand() {
  const input = document.getElementById("cmd-text");
  const text = input.value.trim();
  if (!text) return;
  input.value = "";
  await api("/api/command", { text, robot: selected });
}

function selectRobot(rid, focusInput = true) {
  selected = rid;
  const r = MAP.fleet.find((f) => f.id === rid);
  for (const b of document.querySelectorAll(".talk-tab")) b.classList.toggle("active", b.dataset.id === rid);
  for (const t of document.querySelectorAll(".cam-thumb")) t.classList.toggle("active", t.dataset.id === rid);
  const camLabel = document.getElementById("cam-label");
  if (camLabel) camLabel.textContent = `${rid} ${r.label}`;
  const input = document.getElementById("cmd-text");
  input.placeholder = `對 ${rid}（${r.label}）說…`;
  const presets = document.getElementById("presets");
  presets.innerHTML = "";
  for (const text of [...PRESETS[r.role], ...PRESETS.common]) {
    const b = document.createElement("button");
    b.textContent = text;
    b.onclick = () => { input.value = text; sendCommand(); };
    presets.append(b);
  }
  lastChatKey = "";
  if (STATE) render();
  if (focusInput) input.focus();
}

// ------------------------------------------------------------------ cameras (Webots only)
const MAIN_FPS = 5;
const THUMB_FPS = 1;

function setupCameras() {
  if (!MAP.cameras) return;
  document.getElementById("cam-panel").hidden = false;
  const thumbsOn = document.getElementById("cam-thumbs-on");
  const box = document.getElementById("cam-thumbs");
  for (const r of MAP.fleet) {
    const fig = document.createElement("div");
    fig.className = "cam-thumb";
    fig.dataset.id = r.id;
    fig.innerHTML = `<img alt=""><span>${r.id}</span>`;
    fig.onclick = () => selectRobot(r.id, false);
    box.append(fig);
    // the selected robot is already in the big view, so its thumbnail doesn't fetch twice
    streamCamera(fig.querySelector("img"), () => r.id, THUMB_FPS, () => thumbsOn.checked && selected !== r.id);
  }
  const main = document.getElementById("cam-main");
  let shown = 0, since = performance.now();
  streamCamera(main, () => selected, MAIN_FPS, () => true, (det) => {
    drawDetections(det);
    shown++;
    const dt = (performance.now() - since) / 1000;
    if (dt > 2) {
      document.getElementById("cam-info").textContent = `（${selected} 第一人稱，約 ${(shown / dt).toFixed(1)} 張/秒）`;
      shown = 0; since = performance.now();
    }
  });
}

const DIRT_LABEL = { dry: "乾垃圾", wet: "液體", mixed: "混合髒污" };

// YOLO boxes found in exactly the frame being shown (the server attaches them to the image)
function drawDetections(header) {
  const c = document.getElementById("cam-boxes");
  const tag = document.getElementById("cam-yolo");
  const det = header && header !== "off" ? JSON.parse(header) : null;
  const g = c.getContext("2d");
  if (!det) {
    g.clearRect(0, 0, c.width, c.height);
    tag.textContent = "YOLO 未啟動";
    return;
  }
  const [w, h] = det.size;
  if (c.width !== w * 2) { c.width = w * 2; c.height = h * 2; }
  g.setTransform(2, 0, 0, 2, 0, 0);
  g.clearRect(0, 0, w, h);
  g.font = "bold 9px -apple-system, 'PingFang TC', sans-serif";
  for (const b of det.boxes) {
    const [x1, y1, x2, y2] = b.xyxy;
    const color = DIRT_COLOR[b.cls] || "#fff";
    g.strokeStyle = color; g.lineWidth = 1.5;
    g.strokeRect(x1, y1, x2 - x1, y2 - y1);
    const text = `${DIRT_LABEL[b.cls] || b.cls} ${b.conf.toFixed(2)}`;
    const tw = g.measureText(text).width + 6;
    const ty = y1 >= 12 ? y1 - 12 : y2;
    g.fillStyle = color; g.fillRect(x1, ty, tw, 12);
    g.fillStyle = "#0b0f14"; g.fillText(text, x1 + 3, ty + 9);
  }
  tag.textContent = `YOLO：${det.boxes.length} 個髒污・${det.ms} ms${det.stale ? "（上一張）" : ""}`;
}

// Fetch one JPEG at a time (never piling up requests) at most `fps` times a second.
function streamCamera(img, robotId, fps, active, onFrame) {
  let url = null;
  const tick = async () => {
    const t0 = performance.now();
    if (active()) {
      try {
        const r = await fetch(`/api/camera/${robotId()}.jpg?fps=${fps}`, { cache: "no-store" });
        if (!r.ok) throw new Error(r.status);
        const next = URL.createObjectURL(await r.blob());
        img.src = next;
        if (url) URL.revokeObjectURL(url);
        url = next;
        img.classList.remove("nosignal");
        if (onFrame) onFrame(r.headers.get("X-Detections"));
      } catch (e) {
        img.classList.add("nosignal");
      }
    }
    setTimeout(tick, Math.max(50, 1000 / fps - (performance.now() - t0)));
  };
  tick();
}

function setupMic() {
  // Browser speech recognition stands in for the robot's microphone; the robot's own ASR still
  // "hears" the resulting text, exactly as with typed input.
  const btn = document.getElementById("mic");
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SR) {
    btn.disabled = true;
    btn.title = "這個瀏覽器不支援語音辨識，請用打字輸入";
    return;
  }
  const rec = new SR();
  rec.lang = "zh-TW";
  rec.interimResults = false;
  rec.onresult = (e) => {
    document.getElementById("cmd-text").value = e.results[0][0].transcript;
    sendCommand();
  };
  rec.onend = () => btn.classList.remove("listening");
  rec.onerror = () => btn.classList.remove("listening");
  btn.onclick = () => {
    if (btn.classList.contains("listening")) return rec.stop();
    btn.classList.add("listening");
    rec.start();
  };
}

async function onMapClick(e) {
  const rect = canvas.getBoundingClientRect();
  const fx = ((e.clientX - rect.left) / rect.width) * MAP.w;
  const fy = ((e.clientY - rect.top) / rect.height) * MAP.h;
  const hit = STATE && STATE.robots.find((r) => Math.hypot(r.x + 0.5 - fx, r.y + 0.5 - fy) < 0.8);
  if (hit) return selectRobot(hit.id);
  const tool = document.querySelector("input[name=tool]:checked").value;
  if (tool === "none") return;
  const x = Math.floor(fx);
  const y = Math.floor(fy);
  const [kind, type] = tool.split(":");
  await api("/api/spawn", { kind, type, x, y });
}

async function poll() {
  try {
    STATE = await api("/api/state");
    render();
  } catch (err) {
    console.error(err);
  }
  setTimeout(poll, 150);
}

// ------------------------------------------------------------------ map
function cellRect(x, y, pad = 0) {
  return [x * CS + pad, y * CS + pad, CS - 2 * pad, CS - 2 * pad];
}

function drawMap() {
  ctx.fillStyle = "#1a212c";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  MAP.zones.forEach((z, i) => {
    const [x0, y0, x1, y1] = z.rect;
    ctx.fillStyle = ZONE_TINT[i % ZONE_TINT.length];
    ctx.globalAlpha = 0.55;
    ctx.fillRect(x0 * CS, y0 * CS, (x1 - x0 + 1) * CS, (y1 - y0 + 1) * CS);
    ctx.globalAlpha = 1;
  });

  ctx.fillStyle = "#475468";
  for (const [x, y] of MAP.walls) ctx.fillRect(...cellRect(x, y));

  ctx.strokeStyle = "rgba(255,255,255,0.035)";
  ctx.lineWidth = 1;
  for (let x = 0; x <= MAP.w; x++) { ctx.beginPath(); ctx.moveTo(x * CS + 0.5, 0); ctx.lineTo(x * CS + 0.5, canvas.height); ctx.stroke(); }
  for (let y = 0; y <= MAP.h; y++) { ctx.beginPath(); ctx.moveTo(0, y * CS + 0.5); ctx.lineTo(canvas.width, y * CS + 0.5); ctx.stroke(); }

  ctx.font = "bold 12px -apple-system, 'PingFang TC', sans-serif";
  ctx.fillStyle = "rgba(225,232,242,0.75)";
  MAP.zones.forEach((z) => {
    const [x0, y0, x1] = z.rect;
    if (x1 - x0 < 3) { // narrow aisle: write the name vertically
      [...z.name].forEach((ch, i) => ctx.fillText(ch, x0 * CS + 6, y0 * CS + 30 + i * 14));
    } else {
      ctx.fillText(z.name, x0 * CS + 4, y0 * CS + 13);
    }
  });

  for (const [x, y] of MAP.drop_cells) {
    ctx.strokeStyle = "rgba(242,201,76,0.5)";
    ctx.setLineDash([3, 3]);
    ctx.strokeRect(...cellRect(x, y, 2));
    ctx.setLineDash([]);
  }
  MAP.docks.forEach(([x, y], i) => {
    ctx.fillStyle = "#2e3b2a";
    ctx.fillRect(...cellRect(x, y, 1));
    ctx.fillStyle = "#9be38a";
    ctx.font = "10px sans-serif";
    ctx.fillText("⚡", x * CS + 4, y * CS + 13);
  });
}

function drawState() {
  const s = STATE;
  const showTruth = document.getElementById("show-truth").checked;

  ctx.fillStyle = "rgba(255,170,60,0.28)";
  for (const [x, y] of s.no_go_buffer) ctx.fillRect(...cellRect(x, y));
  ctx.fillStyle = "rgba(255,93,93,0.45)";
  for (const [x, y] of s.no_go_core) ctx.fillRect(...cellRect(x, y));

  for (const d of s.dirt) {
    if (!d.known && !showTruth) continue;
    ctx.globalAlpha = d.known ? 1 : 0.35;
    ctx.fillStyle = DIRT_COLOR[d.kind];
    const cx = d.x * CS + CS / 2, cy = d.y * CS + CS / 2;
    if (d.kind === "wet") {
      ctx.beginPath(); ctx.ellipse(cx, cy, 7, 5, 0, 0, Math.PI * 2); ctx.fill();
    } else if (d.kind === "mixed") {
      ctx.beginPath(); ctx.ellipse(cx, cy, 7, 5, 0, 0, Math.PI * 2); ctx.fill();
      ctx.fillStyle = "#e9d7ff"; ctx.fillRect(cx - 4, cy - 2, 3, 3); ctx.fillRect(cx + 2, cy, 3, 3);
    } else {
      ctx.fillRect(cx - 4, cy - 3, 4, 4); ctx.fillRect(cx + 1, cy + 1, 4, 3); ctx.fillRect(cx - 1, cy - 6, 3, 3);
    }
    if (d.known && d.known !== d.kind) { // the console currently believes something else
      ctx.strokeStyle = "#ff5d5d"; ctx.lineWidth = 1.5; ctx.strokeRect(...cellRect(d.x, d.y, 1));
    }
    ctx.globalAlpha = 1;
  }

  for (const o of s.obstacles) {
    if (!o.known && !showTruth) continue;
    ctx.globalAlpha = o.known ? 1 : 0.35;
    const [x, y, w, h] = cellRect(o.x, o.y, 2);
    if (o.kind === "cable") {
      ctx.strokeStyle = OBS_COLOR.cable; ctx.lineWidth = 2.5;
      ctx.beginPath(); ctx.moveTo(x, y + h / 2);
      for (let i = 1; i <= 4; i++) ctx.lineTo(x + (w * i) / 4, y + (i % 2 ? 2 : h - 2));
      ctx.stroke();
    } else {
      ctx.fillStyle = OBS_COLOR[o.kind] || "#999";
      ctx.fillRect(x, y, w, h);
      if (o.kind === "box") { ctx.strokeStyle = "#7a4515"; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(x, y + h / 2); ctx.lineTo(x + w, y + h / 2); ctx.stroke(); }
    }
    ctx.globalAlpha = 1;
  }

  for (const r of s.robots) {
    if (r.path.length) {
      ctx.strokeStyle = ROLE_COLOR[r.role]; ctx.globalAlpha = 0.45; ctx.lineWidth = 2; ctx.setLineDash([4, 4]);
      ctx.beginPath(); ctx.moveTo(r.x * CS + CS / 2, r.y * CS + CS / 2);
      for (const [x, y] of r.path) ctx.lineTo(x * CS + CS / 2, y * CS + CS / 2);
      ctx.stroke(); ctx.setLineDash([]); ctx.globalAlpha = 1;
      const [nx, ny] = r.path[0]; // reserved next cell
      ctx.strokeStyle = ROLE_COLOR[r.role]; ctx.lineWidth = 1; ctx.strokeRect(...cellRect(nx, ny, 1));
    }
  }
  for (const r of s.robots) {
    const cx = r.x * CS + CS / 2, cy = r.y * CS + CS / 2;
    if (r.id === selected) { // the robot being talked to
      ctx.strokeStyle = "#ffffff"; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(cx, cy, 12, 0, Math.PI * 2); ctx.stroke();
    }
    ctx.fillStyle = "rgba(255,255,255,0.04)";
    ctx.fillRect((r.x - 3) * CS, (r.y - 3) * CS, 7 * CS, 7 * CS); // camera field of view
    ctx.fillStyle = ROLE_COLOR[r.role];
    ctx.beginPath(); ctx.arc(cx, cy, 8, 0, Math.PI * 2); ctx.fill();
    ctx.strokeStyle = r.battery < 25 ? "#ff5d5d" : "#0b0f14";
    ctx.lineWidth = 2;
    ctx.beginPath(); ctx.arc(cx, cy, 8, -Math.PI / 2, -Math.PI / 2 + (Math.PI * 2 * r.battery) / 100); ctx.stroke();
    ctx.fillStyle = "#0b0f14"; ctx.font = "bold 9px sans-serif"; ctx.textAlign = "center";
    ctx.fillText(r.id, cx, cy + 3); ctx.textAlign = "left";
    if (r.carrying) { ctx.fillStyle = OBS_COLOR[r.carrying]; ctx.fillRect(cx + 4, cy - 12, 8, 8); }
  }
}

// ------------------------------------------------------------------ panels
function meter(value, color, text) {
  return `<div class="meter"><div style="width:${Math.max(0, Math.min(100, value))}%;background:${color}"></div><span>${text}</span></div>`;
}

function renderRobots() {
  const tasks = Object.fromEntries(STATE.tasks.map((t) => [t.id, t]));
  document.getElementById("robots").innerHTML = STATE.robots.map((r) => {
    const t = r.task && tasks[r.task];
    const doing = (r.held ? "⏸ 待命中（說「繼續工作」恢復）　" : "") +
      (r.mode === "charge" ? "回充電站" : t ? `${r.task} ${t.title}` : r.mode === "park" ? "停靠" : "");
    const bat = r.battery < 25 ? "#ff5d5d" : "#3ccf8e";
    return `<div class="robot" onclick="selectRobot('${r.id}')" style="cursor:pointer">
      <div class="badge" style="background:${ROLE_COLOR[r.role]}">${r.id}</div>
      <div class="name">${r.label}　<span class="state">${STATE_LABEL[r.state] || r.state}</span></div>
      <div class="meters">${meter(r.battery, bat, `電量 ${r.battery.toFixed(0)}%`)}
        ${r.water !== null ? meter(r.water, "#39b6ff", `水箱 ${r.water.toFixed(0)}%`) : ""}</div>
      <div class="state">${doing}${r.carrying ? `（搬運中）` : ""}</div>
    </div>`;
  }).join("");
}

function renderTasks() {
  const open = STATE.tasks.filter((t) => t.status === "pending" || t.status === "assigned");
  document.getElementById("task-count").textContent = `（進行中/排隊 ${open.length}）`;
  const doneIds = new Set(STATE.tasks.filter((t) => t.status === "done").map((t) => t.id));
  const rows = STATE.tasks.map((t) => {
    const waiting = t.status === "pending" && t.depends_on.length && !t.depends_on.every((d) => doneIds.has(d));
    const cls = waiting ? "waiting" : t.status;
    const label = waiting ? `等 ${t.depends_on.join(",")}` : { pending: "排隊", assigned: "執行中", done: "完成" }[t.status];
    return `<tr><td>${t.id}${t.source === "voice" ? ' <span class="voice">🎤</span>' : ""}</td>
      <td>${t.title}</td><td>${t.assignee || "—"}</td><td><span class="st ${cls}">${label}</span></td></tr>`;
  });
  document.getElementById("tasks").innerHTML =
    `<table><tr><th>ID</th><th>任務</th><th>機器</th><th>狀態</th></tr>${rows.join("")}</table>`;
}

function renderLog() {
  const entries = STATE.log.filter((e) => logFilter.has(e.cat));
  const key = entries.length + ":" + (entries.length ? entries[entries.length - 1].tick + entries[entries.length - 1].text : "");
  if (key === lastLogKey) return;
  lastLogKey = key;
  const box = document.getElementById("log");
  const atBottom = box.scrollHeight - box.scrollTop - box.clientHeight < 30;
  box.innerHTML = entries.map((e) =>
    `<div class="row"><span class="t">${(e.tick * 0.5).toFixed(1).padStart(6)}s</span><span class="c ${e.cat}">${LOG_CATS[e.cat] || e.cat}</span>${escapeHtml(e.text)}</div>`
  ).join("");
  if (atBottom) box.scrollTop = box.scrollHeight;
}

function renderTalk() {
  for (const b of document.querySelectorAll(".talk-tab")) {
    const r = STATE.robots.find((x) => x.id === b.dataset.id);
    const sub = b.querySelector(".sub");
    const label = MAP.fleet.find((f) => f.id === r.id).label;
    sub.textContent = `${label}・${r.held ? "待命中" : STATE_LABEL[r.state] || r.state}`;
    b.classList.toggle("held", r.held);
  }
  const msgs = STATE.chat.filter((m) => m.robot === selected);
  const key = selected + ":" + msgs.length + (msgs.length ? msgs[msgs.length - 1].text : "");
  if (key === lastChatKey) return;
  lastChatKey = key;
  const box = document.getElementById("chat");
  box.innerHTML = msgs.length ? msgs.map((m) =>
    `<div class="bubble ${m.from}"><span class="meta">${m.from === "user" ? `你 → ${m.robot}（機上 ASR 收音）` : m.robot}・${(m.tick * 0.5).toFixed(1)}s</span>${escapeHtml(m.text)}</div>`
  ).join("") : `<div class="empty">還沒對 ${selected} 說過話。<br>指令會由 ${selected} 優先執行；它做不到的部分，中控台會轉給其他機器人。</div>`;
  box.scrollTop = box.scrollHeight;
}

function escapeHtml(s) {
  return s.replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" })[c]);
}

function render() {
  drawMap();
  drawState();
  const s = STATE;
  document.getElementById("clock").textContent =
    `t = ${s.time_s.toFixed(1)} s（tick ${s.tick}）${s.paused ? "・已暫停" : ""}${s.held.length === s.robots.length ? "・全體待命" : ""}${s.thinking ? "・LLM 規劃中…" : ""}`;
  document.getElementById("backends").innerHTML =
    `<span>規劃器 ${s.backends.planner}</span><span>ASR ${s.backends.asr}</span><span>偵測 ${s.backends.detector}</span>`;
  document.getElementById("btn-pause").textContent = s.paused ? "繼續" : "暫停";
  const st = s.stats;
  document.getElementById("stats").innerHTML =
    `場上髒污 <b>${st.dirt_left}</b>（中控台已知 <b>${st.dirt_known}</b>）・已清除 <b>${st.cleaned}</b> 格・` +
    `搬走障礙 <b>${st.obstacles_cleared}</b>・碰撞 <b>${st.collisions}</b>・訊息 <b>${st.messages}</b> 則・避讓物件：${s.avoid.join(", ")}`;
  renderRobots();
  renderTalk();
  renderTasks();
  renderLog();
}

init();
