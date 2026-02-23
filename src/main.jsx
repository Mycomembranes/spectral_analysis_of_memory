import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import WeightDynamicsDashboard from "../weight_dynamics_dashboard.jsx";

createRoot(document.getElementById("root")).render(
  <StrictMode>
    <WeightDynamicsDashboard />
  </StrictMode>
);
