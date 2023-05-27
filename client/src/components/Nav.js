import React from "react";
import { Link } from "react-router-dom";
import "../assets/styles/navbar.css";

export default function Nav() {
  return (
    <nav
      className="navbar flex justify-space-betn"
      style={{ alignItems: "center" }}
    >
      <ul>
        <div className="flex center">
          <li>
            <Link to="/">
              <h1 className="logo">TextWiz</h1>
            </Link>
          </li>
        </div>

        <li className="absolute absolute-center">
          <h2>Summarizer</h2>
        </li>

        <div className="flex center">
          <li>
            <Link to="/about">About</Link>
          </li>
        </div>
      </ul>
    </nav>
  );
}
