import Nav from "./components/Nav";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import "./assets/styles/App.css";
import Home from "./components/Home";

function App() {
  return (
    <>
      <BrowserRouter>
        <Nav />
        <Routes>
          <Route path="/" element={<Home />} />
        </Routes>
      </BrowserRouter>
    </>
  );
}

export default App;
