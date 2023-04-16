
function setSQLConnection() {
  const modal = document.getElementById("sqlmodal");
  
  modal.classList.toggle("hidden");

}
const configForm = document.getElementById("config-form");
if (configForm) {
  configForm.addEventListener("submit", function(event) {
    event.preventDefault();

    localStorage.setItem("sqlserver", document.getElementById("sqlserver").value);
    localStorage.setItem("database", document.getElementById("database").value);
    localStorage.setItem("username", document.getElementById("username").value);
    localStorage.setItem("password", document.getElementById("password").value);
    localStorage.setItem("openai_base", document.getElementById("openai_base").value);
    localStorage.setItem("openai_key", document.getElementById("openai_key").value);
    localStorage.setItem("openai_deployment", document.getElementById("openai_deployment").value);
    
    const modal = document.getElementById("sqlmodal");

    modal.classList.toggle("hidden");
  });
}
