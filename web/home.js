const params = new URLSearchParams(location.search);
if (params.has("access_token")) {
  params.delete("access_token");
  history.replaceState({}, "", location.pathname + (params.size ? "?" + params : ""));
}

document.getElementById("createRoom").onclick = () => {
  location.href = "/room/new";
};
document.getElementById("viewRooms").onclick = () => {
  location.href = "/rooms";
};
