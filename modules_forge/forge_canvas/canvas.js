class TextAreaBinder {
  /**
   * Constructor for TextAreaBinder class.
   * @param {string} elementId - The ID of the element.
   * @param {string} classNamePrefix - The prefix of the class name.
   */
  constructor(elementId, classNamePrefix) {
    // Selects the textarea element within the specified element ID and class prefix.
    this.target = document.querySelector(
      "#" + elementId + "." + classNamePrefix + " textarea",
    );
    // Flag to prevent infinite loop during value synchronization.
    this.sync_lock = false;
    // Stores the previous value of the textarea to detect changes.
    this.previousValue = "";
  }

  /**
   * Sets the value of the textarea element.
   * @param {string} newValue - The new value to set.
   */
  setValue(newValue) {
    // If synchronization is locked, return to avoid infinite loops.
    if (this.sync_lock) {
      return;
    }
    // Lock synchronization to prevent feedback loop.
    this.sync_lock = true;
    // Set the value of the textarea element.
    this.target.value = newValue;
    // Update the previous value.
    this.previousValue = newValue;
    // Create a new 'input' event to simulate user input.
    let inputEvent = new Event("input", { bubbles: true });
    // Define the 'target' property of the event to be the textarea element.
    Object.defineProperty(inputEvent, "target", { value: this.target });
    // Dispatch the 'input' event on the textarea element to trigger any listeners.
    this.target.dispatchEvent(inputEvent);
    // Update the previous value again, might be redundant.
    this.previousValue = newValue;
    // Unlock synchronization.
    this.sync_lock = false;
  }

  /**
   * Listens for changes in the textarea value at a fixed interval.
   * @param {function} callback - The callback function to execute when the value changes.
   */
  listen(callback) {
    // Set interval to check for changes every 100 milliseconds.
    setInterval(() => {
      // If the current value is different from the previous value, it has changed.
      if (this.target.value !== this.previousValue) {
        // Update the previous value to the current value.
        this.previousValue = this.target.value;
        // If synchronization is locked, return to avoid recursion.
        if (this.sync_lock) {
          return;
        }
        // Lock synchronization before executing the callback.
        this.sync_lock = true;
        // Execute the callback function with the current textarea value.
        callback(this.target.value);
        // Unlock synchronization after callback execution.
        this.sync_lock = false;
      }
    }, 100);
  }
}

class ForgeCanvas {
  /**
   * Constructor for ForgeCanvas class, handling image manipulation on a canvas.
   * @param {string} uuid - Unique identifier for the canvas instance.
   * @param {boolean} noUpload - Disables image upload functionality if true.
   * @param {boolean} noScribbles - Disables scribble functionality if true.
   * @param {boolean} contrastScribbles - Enables contrast scribble mode if true.
   * @param {number} initialHeight - Initial height of the canvas container.
   * @param {string} defaultScribbleColor - Default color for scribbles.
   * @param {boolean} lockColor - Locks scribble color if true.
   * @param {number} defaultScribbleWidth - Default width for scribbles.
   * @param {boolean} lockWidth - Locks scribble width if true.
   * @param {number} defaultScribbleAlpha - Default alpha (opacity) for scribbles.
   * @param {boolean} lockAlpha - Locks scribble alpha if true.
   * @param {number} defaultScribbleSoftness - Default softness for scribbles.
   * @param {boolean} lockSoftness - Locks scribble softness if true.
   */
  constructor(
    uuid,
    noUpload = false,
    noScribbles = false,
    contrastScribbles = false,
    initialHeight = 512,
    defaultScribbleColor = "#000000",
    lockColor = false,
    defaultScribbleWidth = 4,
    lockWidth = false,
    defaultScribbleAlpha = 100,
    lockAlpha = false,
    defaultScribbleSoftness = 0,
    lockSoftness = false,
  ) {
    // Configuration from gradio (likely related to UI framework).
    this.gradio_config = gradio_config;
    // Unique identifier for this canvas instance.
    this.uuid = uuid;
    // Disable scribble functionality if true.
    this.no_scribbles = noScribbles;
    // Use contrast scribbles for better visibility if true.
    this.contrast_scribbles = contrastScribbles;
    // Disable image upload functionality if true.
    this.no_upload = noUpload;
    // Initial height of the canvas container.
    this.initial_height = initialHeight;
    // Image object to hold the loaded image.
    this.img = null;
    // X-coordinate of the image position on the canvas.
    this.imgX = 0;
    // Y-coordinate of the image position on the canvas.
    this.imgY = 0;
    // Original width of the loaded image.
    this.orgWidth = 0;
    // Original height of the loaded image.
    this.orgHeight = 0;
    // Scaling factor for the image.
    this.imgScale = 1;
    // Flag indicating if dragging is in progress.
    this.dragging = false;
    // Flag to prevent immediate context menu after dragging.
    this.dragged_just_now = false;
    // Flag indicating if resizing is in progress.
    this.resizing = false;
    // Flag indicating if drawing is in progress.
    this.drawing = false;
    // Current scribble color.
    this.scribbleColor = defaultScribbleColor;
    // Current scribble width.
    this.scribbleWidth = defaultScribbleWidth;
    // Current scribble alpha (opacity).
    this.scribbleAlpha = defaultScribbleAlpha;
    // Current scribble softness/blur.
    this.scribbleSoftness = defaultScribbleSoftness;
    // Lock scribble color if true.
    this.scribbleColorFixed = lockColor;
    // Lock scribble width if true.
    this.scribbleWidthFixed = lockWidth;
    // Lock scribble alpha if true.
    this.scribbleAlphaFixed = lockAlpha;
    // Lock scribble softness if true.
    this.scribbleSoftnessFixed = lockSoftness;
    // History of canvas states for undo/redo functionality.
    this.history = [];
    // Index of the current state in the history.
    this.historyIndex = -1;
    // Flag indicating if canvas is maximized.
    this.maximized = false;
    // Stores original state before maximization.
    this.originalState = {};
    // Pattern for contrast scribbles.
    this.contrast_pattern = null;
    // Flag indicating if pointer is inside the canvas container.
    this.pointerInsideContainer = false;
    // Temporary canvas for drawing operations.
    this.temp_canvas = document.createElement("canvas");
    // Array to store points for drawing path.
    this.temp_draw_points = [];
    // Background image data for temporary drawing.
    this.temp_draw_bg = null;
    // Text area binder for background image data (likely for gradio integration).
    this.background_gradio_bind = new TextAreaBinder(
      this.uuid,
      "logical_image_background",
    );
    // Text area binder for foreground (scribble) image data.
    this.foreground_gradio_bind = new TextAreaBinder(
      this.uuid,
      "logical_image_foreground",
    );
    this.panMode = this.no_scribbles; // Default to pan mode if scribbles disabled
    this.lastTouchDistance = 0; // For pinch zoom
    this.touching = false; // Track touch state
    // Initialize the canvas and event listeners.
    this.start();
  }

  /**
   * Initializes the canvas and sets up event listeners for user interactions.
   */
  start() {
    let forgeCanvas = this;
    // Get references to all necessary HTML elements using their IDs.
    const imageContainerElement = document.getElementById(
        "imageContainer_" + forgeCanvas.uuid,
      ),
      imageElement = document.getElementById("image_" + forgeCanvas.uuid),
      resizeLineElement = document.getElementById(
        "resizeLine_" + forgeCanvas.uuid,
      ),
      containerElement = document.getElementById(
        "container_" + forgeCanvas.uuid,
      ),
      toolbarElement = document.getElementById("toolbar_" + forgeCanvas.uuid),
      uploadButtonElement = document.getElementById(
        "uploadButton_" + forgeCanvas.uuid,
      ),
      resetButtonElement = document.getElementById(
        "resetButton_" + forgeCanvas.uuid,
      ),
      centerImageButtonElement = document.getElementById(
        "centerButton_" + forgeCanvas.uuid,
      ),
      removeButtonElement = document.getElementById(
        "removeButton_" + forgeCanvas.uuid,
      ),
      undoButtonElement = document.getElementById(
        "undoButton_" + forgeCanvas.uuid,
      ),
      redoButtonElement = document.getElementById(
        "redoButton_" + forgeCanvas.uuid,
      ),
      drawingCanvasElement = document.getElementById(
        "drawingCanvas_" + forgeCanvas.uuid,
      ),
      maximizeButtonElement = document.getElementById(
        "maxButton_" + forgeCanvas.uuid,
      ),
      minimizeButtonElement = document.getElementById(
        "minButton_" + forgeCanvas.uuid,
      ),
      scribbleIndicatorElement = document.getElementById(
        "scribbleIndicator_" + forgeCanvas.uuid,
      ),
      uploadHintElement = document.getElementById(
        "uploadHint_" + forgeCanvas.uuid,
      ),
      scribbleColorElement = document.getElementById(
        "scribbleColor_" + forgeCanvas.uuid,
      ),
      scribbleColorBlockElement = document.getElementById(
        "scribbleColorBlock_" + forgeCanvas.uuid,
      ),
      scribbleWidthElement = document.getElementById(
        "scribbleWidth_" + forgeCanvas.uuid,
      ),
      widthLabelElement = document.getElementById(
        "widthLabel_" + forgeCanvas.uuid,
      ),
      scribbleWidthBlockElement = document.getElementById(
        "scribbleWidthBlock_" + forgeCanvas.uuid,
      ),
      scribbleAlphaElement = document.getElementById(
        "scribbleAlpha_" + forgeCanvas.uuid,
      ),
      alphaLabelElement = document.getElementById(
        "alphaLabel_" + forgeCanvas.uuid,
      ),
      scribbleAlphaBlockElement = document.getElementById(
        "scribbleAlphaBlock_" + forgeCanvas.uuid,
      ),
      scribbleSoftnessElement = document.getElementById(
        "scribbleSoftness_" + forgeCanvas.uuid,
      ),
      softnessLabelElement = document.getElementById(
        "softnessLabel_" + forgeCanvas.uuid,
      ),
      scribbleSoftnessBlockElement = document.getElementById(
        "scribbleSoftnessBlock_" + forgeCanvas.uuid,
      ),
      panModeButtonElement = document.getElementById(
        "panModeButton_" + forgeCanvas.uuid,
      );

    // Initialize scribble color, width, alpha, and softness input values.
    scribbleColorElement.value = forgeCanvas.scribbleColor;
    scribbleWidthElement.value = forgeCanvas.scribbleWidth;
    scribbleAlphaElement.value = forgeCanvas.scribbleAlpha;
    scribbleSoftnessElement.value = forgeCanvas.scribbleSoftness;

    // Set initial size of scribble indicator.
    const indicatorSize = forgeCanvas.scribbleWidth * 20;
    scribbleIndicatorElement.style.width = indicatorSize + "px";
    scribbleIndicatorElement.style.height = indicatorSize + "px";

    const touchDevice =
      navigator.maxTouchPoints || "ontouchstart" in document.documentElement;
    var lockToolbar = false;
    if (touchDevice && touchDevice >= 1) {
      toolbarElement.style.opacity = "1";
      lockToolbar = true;
    }

    // Set initial height of the container.
    containerElement.style.height = forgeCanvas.initial_height + "px";

    // Set canvas dimensions to the container's client dimensions.
    drawingCanvasElement.width = imageContainerElement.clientWidth;
    drawingCanvasElement.height = imageContainerElement.clientHeight;

    // Get 2D rendering context of the drawing canvas.
    const ctx = drawingCanvasElement.getContext("2d");
    forgeCanvas.drawingCanvas = drawingCanvasElement;

    // Hide scribble related elements if scribbling is disabled.
    if (forgeCanvas.no_scribbles) {
      panModeButtonElement.style.display = "none";
      resetButtonElement.style.display = "none";
      undoButtonElement.style.display = "none";
      redoButtonElement.style.display = "none";
      scribbleColorElement.style.display = "none";
      scribbleColorBlockElement.style.display = "none";
      scribbleWidthBlockElement.style.display = "none";
      scribbleAlphaBlockElement.style.display = "none";
      scribbleSoftnessBlockElement.style.display = "none";
      scribbleIndicatorElement.style.display = "none";
      drawingCanvasElement.style.display = "none";
    }

    // Hide upload related elements if upload is disabled.
    if (forgeCanvas.no_upload) {
      uploadButtonElement.style.display = "none";
      uploadHintElement.style.display = "none";
    }

    // Configure contrast scribbles if enabled.
    if (forgeCanvas.contrast_scribbles) {
      scribbleColorBlockElement.style.display = "none";
      scribbleAlphaBlockElement.style.display = "none";
      scribbleSoftnessBlockElement.style.display = "none";
      const patternCanvas = forgeCanvas.temp_canvas;
      patternCanvas.width = 20;
      patternCanvas.height = 20;
      const patternCtx = patternCanvas.getContext("2d");
      patternCtx.fillStyle = "#ffffff";
      patternCtx.fillRect(0, 0, 10, 10);
      patternCtx.fillRect(10, 10, 10, 10);
      patternCtx.fillStyle = "#000000";
      patternCtx.fillRect(10, 0, 10, 10);
      patternCtx.fillRect(0, 10, 10, 10);
      forgeCanvas.contrast_pattern = ctx.createPattern(patternCanvas, "repeat");
      drawingCanvasElement.style.opacity = "0.5"; // Make canvas slightly transparent for contrast scribbles
    }

    // Adjust layout if scribble properties are fixed (locked).
    if (
      forgeCanvas.contrast_scribbles ||
      (forgeCanvas.scribbleColorFixed &&
        forgeCanvas.scribbleAlphaFixed &&
        forgeCanvas.scribbleSoftnessFixed)
    ) {
      scribbleWidthBlockElement.style.width = "100%";
      scribbleWidthElement.style.width = "100%";
      widthLabelElement.style.display = "none";
    }
    if (forgeCanvas.scribbleColorFixed)
      scribbleColorBlockElement.style.display = "none";
    if (forgeCanvas.scribbleWidthFixed)
      scribbleWidthBlockElement.style.display = "none";
    if (forgeCanvas.scribbleAlphaFixed)
      scribbleAlphaBlockElement.style.display = "none";
    if (forgeCanvas.scribbleSoftnessFixed)
      scribbleSoftnessBlockElement.style.display = "none";

    // Observe container resize to adjust image and canvas.
    const resizeObserver = new ResizeObserver(() => {
      forgeCanvas.adjustInitialPositionAndScale();
      forgeCanvas.drawImage();
    });
    resizeObserver.observe(containerElement);

    // Event listener for file input change (image upload).
    document
      .getElementById("imageInput_" + forgeCanvas.uuid)
      .addEventListener("change", function (event) {
        forgeCanvas.handleFileUpload(event.target.files[0]);
      });

    // Event listener for upload button click (triggers file input click).
    uploadButtonElement.addEventListener("click", function () {
      if (forgeCanvas.no_upload) {
        return;
      }
      document.getElementById("imageInput_" + forgeCanvas.uuid).click();
    });

    // Event listener for reset button click (resets image and scribbles).
    resetButtonElement.addEventListener("click", function () {
      forgeCanvas.resetImage();
    });

    // Event listener for center image button click (centers the image).
    centerImageButtonElement.addEventListener("click", function () {
      forgeCanvas.adjustInitialPositionAndScale();
      forgeCanvas.drawImage();
    });

    // Event listener for remove image button click (removes the image).
    removeButtonElement.addEventListener("click", function () {
      forgeCanvas.removeImage();
    });

    // Event listener for undo button click.
    undoButtonElement.addEventListener("click", function () {
      forgeCanvas.undo();
    });

    // Event listener for redo button click.
    redoButtonElement.addEventListener("click", function () {
      forgeCanvas.redo();
    });

    // Event listener for scribble color input change.
    scribbleColorElement.addEventListener("input", function () {
      forgeCanvas.scribbleColor = this.value;
      scribbleIndicatorElement.style.borderColor = forgeCanvas.scribbleColor;
    });

    // Event listener for scribble width input change.
    scribbleWidthElement.addEventListener("input", function () {
      forgeCanvas.scribbleWidth = this.value;
      const indicatorSize = forgeCanvas.scribbleWidth * 20;
      scribbleIndicatorElement.style.width = indicatorSize + "px";
      scribbleIndicatorElement.style.height = indicatorSize + "px";
    });

    // Event listener for scribble alpha input change.
    scribbleAlphaElement.addEventListener("input", function () {
      forgeCanvas.scribbleAlpha = this.value;
    });

    // Event listener for scribble softness input change.
    scribbleSoftnessElement.addEventListener("input", function () {
      forgeCanvas.scribbleSoftness = this.value;
    });

    // Event listener for pointer down on drawing canvas (start drawing).
    drawingCanvasElement.addEventListener("pointerdown", function (event) {
      if (
        !forgeCanvas.img || // No image loaded
        event.button !== 0 || // Not left mouse button
        forgeCanvas.no_scribbles // Scribbles disabled
      ) {
        return;
      }
      if (forgeCanvas.panMode) return;
      const canvasRect = drawingCanvasElement.getBoundingClientRect();
      forgeCanvas.drawing = true;
      drawingCanvasElement.style.cursor = "crosshair";
      scribbleIndicatorElement.style.display = "none";
      forgeCanvas.temp_draw_points = [
        [
          (event.clientX - canvasRect.left) / forgeCanvas.imgScale,
          (event.clientY - canvasRect.top) / forgeCanvas.imgScale,
        ],
      ];
      forgeCanvas.temp_draw_bg = ctx.getImageData(
        0,
        0,
        drawingCanvasElement.width,
        drawingCanvasElement.height,
      );
      forgeCanvas.handleDraw(event);
    });

    // Event listener for pointer move on drawing canvas (continue drawing).
    drawingCanvasElement.addEventListener("pointermove", function (event) {
      if (forgeCanvas.panMode) return;
      if (forgeCanvas.drawing) forgeCanvas.handleDraw(event);
      if (forgeCanvas.img && !forgeCanvas.dragging)
        drawingCanvasElement.style.cursor = "crosshair";

      // Show scribble indicator when mouse moves over canvas and not drawing/dragging
      if (
        forgeCanvas.img &&
        !forgeCanvas.drawing &&
        !forgeCanvas.dragging &&
        !forgeCanvas.no_scribbles
      ) {
        const containerRect = imageContainerElement.getBoundingClientRect();
        const indicatorOffset = forgeCanvas.scribbleWidth * 10;
        scribbleIndicatorElement.style.left =
          event.clientX - containerRect.left - indicatorOffset + "px";
        scribbleIndicatorElement.style.top =
          event.clientY - containerRect.top - indicatorOffset + "px";
        scribbleIndicatorElement.style.display = "block";
      }
    });

    // Event listener for pointer up on drawing canvas (stop drawing).
    drawingCanvasElement.addEventListener("pointerup", function () {
      forgeCanvas.drawing = false;
      drawingCanvasElement.style.cursor = "";
      forgeCanvas.saveState();
    });

    // Event listener for pointer leave drawing canvas (stop drawing, hide indicator).
    drawingCanvasElement.addEventListener("pointerleave", function () {
      forgeCanvas.drawing = false;
      drawingCanvasElement.style.cursor = "";
      scribbleIndicatorElement.style.display = "none";
    });

    // Event listener for pointer move on image container (dragging).
    imageContainerElement.addEventListener("pointermove", function (event) {
      if (forgeCanvas.dragging) {
        const containerRect = imageContainerElement.getBoundingClientRect();
        const mouseX = event.clientX - containerRect.left;
        const mouseY = event.clientY - containerRect.top;
        forgeCanvas.imgX = mouseX - forgeCanvas.offsetX;
        forgeCanvas.imgY = mouseY - forgeCanvas.offsetY;
        forgeCanvas.drawImage();
        forgeCanvas.dragged_just_now = true; // Flag drag for context menu prevention
      }
    });

    // Event listener for pointer up on image container (stop dragging).
    imageContainerElement.addEventListener("pointerup", function (event) {
      if (forgeCanvas.dragging) forgeCanvas.handleDragEnd(event, false);
    });

    // Event listener for pointer leave image container (stop dragging).
    imageContainerElement.addEventListener("pointerleave", function (event) {
      if (forgeCanvas.dragging) forgeCanvas.handleDragEnd(event, true);
    });

    // Event listener for mouse wheel on image container (zoom).
    imageContainerElement.addEventListener("wheel", function (event) {
      if (!forgeCanvas.img) {
        return;
      }
      event.preventDefault(); // Prevent page scroll
      const containerRect = imageContainerElement.getBoundingClientRect();
      const mouseX = event.clientX - containerRect.left;
      const mouseY = event.clientY - containerRect.top;
      const oldScale = forgeCanvas.imgScale;
      const scaleFactor = event.deltaY * -0.001; // Adjust scale speed
      forgeCanvas.imgScale += scaleFactor;
      forgeCanvas.imgScale = Math.max(0.1, forgeCanvas.imgScale); // Limit min scale
      const scaleRatio = forgeCanvas.imgScale / oldScale;
      forgeCanvas.imgX = mouseX - (mouseX - forgeCanvas.imgX) * scaleRatio;
      forgeCanvas.imgY = mouseY - (mouseY - forgeCanvas.imgY) * scaleRatio;
      forgeCanvas.drawImage();
    });

    // Prevent context menu on right click after dragging (to allow drag pan).
    imageContainerElement.addEventListener("contextmenu", function (event) {
      if (forgeCanvas.dragged_just_now) event.preventDefault();
      forgeCanvas.dragged_just_now = false;
    });

    // Show toolbar on pointer over image container.
    imageContainerElement.addEventListener("pointerover", function () {
      toolbarElement.style.opacity = "1";
      if (!forgeCanvas.img && !forgeCanvas.no_upload)
        imageContainerElement.style.cursor = "pointer"; // Hint upload on empty canvas
    });

    // Hide toolbar and reset cursors on pointer out of image container.
    imageContainerElement.addEventListener("pointerout", function () {
      if (!lockToolbar) toolbarElement.style.opacity = "0";
      imageElement.style.cursor = "";
      drawingCanvasElement.style.cursor = "";
      imageContainerElement.style.cursor = "";
      scribbleIndicatorElement.style.display = "none";
    });

    // Event listener for resize line pointer down (start resizing container).
    resizeLineElement.addEventListener("pointerdown", function (event) {
      forgeCanvas.resizing = true;
      event.preventDefault();
      event.stopPropagation();
    });

    // Event listener for document pointer move (resizing container).
    document.addEventListener("pointermove", function (event) {
      if (forgeCanvas.resizing) {
        const containerRect = containerElement.getBoundingClientRect();
        const mouseY = event.clientY - containerRect.top;
        containerElement.style.height = mouseY + "px";
        event.preventDefault();
        event.stopPropagation();
      }
    });

    // Event listeners to stop resizing on pointer up/leave document.
    document.addEventListener("pointerup", function () {
      forgeCanvas.resizing = false;
    });
    document.addEventListener("pointerleave", function (event) {
      forgeCanvas.resizing = false;
    });

    // Prevent default drag behaviors on image container.
    ["dragenter", "dragover", "dragleave", "drop"].forEach((eventType) => {
      imageContainerElement.addEventListener(
        eventType,
        preventDefaultEvent,
        false,
      );
    });
    function preventDefaultEvent(event) {
      event.preventDefault();
      event.stopPropagation();
    }

    // Change cursor on drag enter image container.
    imageContainerElement.addEventListener("dragenter", () => {
      imageElement.style.cursor = "copy";
      drawingCanvasElement.style.cursor = "copy";
    });

    // Reset cursor on drag leave image container.
    imageContainerElement.addEventListener("dragleave", () => {
      imageElement.style.cursor = "";
      drawingCanvasElement.style.cursor = "";
    });

    // Handle image drop on image container.
    imageContainerElement.addEventListener("drop", function (event) {
      imageElement.style.cursor = "";
      drawingCanvasElement.style.cursor = "";
      const dataTransfer = event.dataTransfer;
      const files = dataTransfer.files;
      if (files.length > 0) forgeCanvas.handleFileUpload(files[0]);
    });

    // Track pointer inside container for paste event handling.
    imageContainerElement.addEventListener("pointerenter", () => {
      forgeCanvas.pointerInsideContainer = true;
    });
    imageContainerElement.addEventListener("pointerleave", () => {
      forgeCanvas.pointerInsideContainer = false;
    });

    // Handle paste event on document (only if pointer is inside container).
    document.addEventListener("paste", function (event) {
      if (forgeCanvas.pointerInsideContainer) forgeCanvas.handlePaste(event);
    });

    // Handle keyboard shortcuts (Ctrl+Z for undo, Ctrl+Y for redo).
    document.addEventListener("keydown", (event) => {
      if (!forgeCanvas.pointerInsideContainer) {
        return;
      }
      if (event.ctrlKey && event.key === "z") {
        event.preventDefault();
        this.undo();
      }
      if (event.ctrlKey && event.key === "y") {
        event.preventDefault();
        this.redo();
      }
    });

    // Event listener for maximize button click.
    maximizeButtonElement.addEventListener("click", function () {
      forgeCanvas.maximize();
    });

    // Event listener for minimize button click.
    minimizeButtonElement.addEventListener("click", function () {
      forgeCanvas.minimize();
    });

    // Update pan mode button appearance
    const updatePanModeButton = () => {
      panModeButtonElement.textContent = !forgeCanvas.panMode ? "✏️" : "✋";
      panModeButtonElement.title = forgeCanvas.panMode
        ? "Draw Mode"
        : "Pan Mode";
    };
    updatePanModeButton();

    // Pan mode toggle button click handler
    panModeButtonElement.addEventListener("click", function () {
      forgeCanvas.panMode = !forgeCanvas.panMode;
      updatePanModeButton();
    });

    // Pointer/touch event handlers
    imageContainerElement.addEventListener("pointerdown", function (event) {
      const containerRect = imageContainerElement.getBoundingClientRect();
      const offsetX = event.clientX - containerRect.left;
      const offsetY = event.clientY - containerRect.top;

      // Start dragging if in pan mode or right click
      if ((forgeCanvas.panMode && event.button === 0) || event.button === 2) {
        if (forgeCanvas.isInsideImage(offsetX, offsetY)) {
          forgeCanvas.dragging = true;
          forgeCanvas.offsetX = offsetX - forgeCanvas.imgX;
          forgeCanvas.offsetY = offsetY - forgeCanvas.imgY;
          imageElement.style.cursor = "grabbing";
          drawingCanvasElement.style.cursor = "grabbing";
          scribbleIndicatorElement.style.display = "none";
        }
      } else if (
        event.button === 0 &&
        !forgeCanvas.img &&
        !forgeCanvas.no_upload
      ) {
        document.getElementById("imageInput_" + forgeCanvas.uuid).click();
      }
    });

    // Touch event handlers
    imageContainerElement.addEventListener("touchstart", function (event) {
      if (event.target.closest(".forge-toolbar")) return; // Don't handle touches on toolbar
      if (!forgeCanvas.panMode) return;

      event.preventDefault();
      forgeCanvas.touching = true;

      if (event.touches.length === 1) {
        const touch = event.touches[0];
        const containerRect = imageContainerElement.getBoundingClientRect();
        forgeCanvas.offsetX =
          touch.clientX - containerRect.left - forgeCanvas.imgX;
        forgeCanvas.offsetY =
          touch.clientY - containerRect.top - forgeCanvas.imgY;
      } else if (event.touches.length === 2) {
        // Store initial pinch distance
        const touch1 = event.touches[0];
        const touch2 = event.touches[1];
        forgeCanvas.lastTouchDistance = Math.hypot(
          touch2.clientX - touch1.clientX,
          touch2.clientY - touch1.clientY,
        );
      }
    });

    imageContainerElement.addEventListener("touchmove", function (event) {
      if (!forgeCanvas.panMode) return;
      if (!forgeCanvas.touching) return;
      event.preventDefault();

      if (event.touches.length === 1) {
        // Pan
        const touch = event.touches[0];
        const containerRect = imageContainerElement.getBoundingClientRect();
        forgeCanvas.imgX =
          touch.clientX - containerRect.left - forgeCanvas.offsetX;
        forgeCanvas.imgY =
          touch.clientY - containerRect.top - forgeCanvas.offsetY;
        forgeCanvas.drawImage();
      } else if (event.touches.length === 2) {
        // Pinch zoom
        const touch1 = event.touches[0];
        const touch2 = event.touches[1];
        const currentDistance = Math.hypot(
          touch2.clientX - touch1.clientX,
          touch2.clientY - touch1.clientY,
        );

        const scale = currentDistance / forgeCanvas.lastTouchDistance;
        const oldScale = forgeCanvas.imgScale;
        forgeCanvas.imgScale *= scale;
        forgeCanvas.imgScale = Math.max(0.1, forgeCanvas.imgScale);

        // Adjust position to zoom toward center of pinch
        const centerX = (touch1.clientX + touch2.clientX) / 2;
        const centerY = (touch1.clientY + touch2.clientY) / 2;
        const containerRect = imageContainerElement.getBoundingClientRect();
        const scaleRatio = forgeCanvas.imgScale / oldScale;
        forgeCanvas.imgX =
          centerX -
          containerRect.left -
          (centerX - containerRect.left - forgeCanvas.imgX) * scaleRatio;
        forgeCanvas.imgY =
          centerY -
          containerRect.top -
          (centerY - containerRect.top - forgeCanvas.imgY) * scaleRatio;

        forgeCanvas.lastTouchDistance = currentDistance;
        forgeCanvas.drawImage();
      }
    });

    imageContainerElement.addEventListener("touchend", function () {
      if (!forgeCanvas.panMode) return;
      forgeCanvas.touching = false;
    });

    // Initialize undo/redo button states.
    forgeCanvas.updateUndoRedoButtons();

    // Listen for changes in background image textarea (for gradio integration).
    forgeCanvas.background_gradio_bind.listen(function (base64Image) {
      forgeCanvas.uploadBase64(base64Image);
    });

    // Listen for changes in foreground drawing canvas textarea (for gradio).
    forgeCanvas.foreground_gradio_bind.listen(function (base64DrawingCanvas) {
      forgeCanvas.uploadBase64DrawingCanvas(base64DrawingCanvas);
    });
  }

  /**
   * Handles drawing on the canvas, drawing lines based on pointer movements.
   * @param {PointerEvent} event - Pointer event object.
   */
  handleDraw(event) {
    const canvas = this.drawingCanvas,
      ctx = canvas.getContext("2d"),
      canvasRect = canvas.getBoundingClientRect();
    const x = (event.clientX - canvasRect.left) / this.imgScale;
    const y = (event.clientY - canvasRect.top) / this.imgScale;
    this.temp_draw_points.push([x, y]);

    ctx.putImageData(this.temp_draw_bg, 0, 0); // Restore background

    ctx.beginPath();
    ctx.moveTo(this.temp_draw_points[0][0], this.temp_draw_points[0][1]); // Start from first point
    for (
      let i = 1;
      i < this.temp_draw_points.length;
      i++ // Draw line to each point
    ) {
      ctx.lineTo(this.temp_draw_points[i][0], this.temp_draw_points[i][1]);
    }

    ctx.lineCap = "round"; // Rounded line ends
    ctx.lineJoin = "round"; // Rounded line joins
    ctx.lineWidth = (this.scribbleWidth / this.imgScale) * 20; // Scale line width

    // Handle contrast scribbles
    if (this.contrast_scribbles) {
      ctx.strokeStyle = this.contrast_pattern;
      ctx.stroke();
      return;
    }

    ctx.strokeStyle = this.scribbleColor; // Set scribble color

    // Handle alpha=0 as eraser
    if (!(this.scribbleAlpha > 0)) {
      ctx.globalCompositeOperation = "destination-out"; // Erase mode
      ctx.globalAlpha = 1;
      ctx.stroke();
      return;
    }

    ctx.globalCompositeOperation = "source-over"; // Default draw mode

    // Handle softness (blur effect)
    if (!(this.scribbleSoftness > 0)) {
      ctx.globalAlpha = this.scribbleAlpha / 100; // Set alpha
      ctx.stroke();
      return;
    }

    // Soft brush effect (multiple strokes with varying width and alpha)
    const innerLineWidth = ctx.lineWidth * (1 - this.scribbleSoftness / 150);
    const outerLineWidth = ctx.lineWidth * (1 + this.scribbleSoftness / 150);
    const steps = Math.round(5 + this.scribbleSoftness / 5);
    const lineWidthIncrement = (outerLineWidth - innerLineWidth) / (steps - 1);
    ctx.globalAlpha =
      1 - Math.pow(1 - Math.min(this.scribbleAlpha / 100, 0.95), 1 / steps); // Adjust alpha for softness
    for (let step = 0; step < steps; step++) {
      ctx.lineWidth = innerLineWidth + lineWidthIncrement * step;
      ctx.stroke();
    }
  }

  /**
   * Handles file upload, reads file as Data URL and uploads as base64.
   * @param {File} file - The uploaded file.
   */
  handleFileUpload(file) {
    if (file && !this.no_upload) {
      const reader = new FileReader();
      reader.onload = (event) => {
        this.uploadBase64(event.target.result); // Upload base64 data when loaded
      };
      reader.readAsDataURL(file); // Read file as Data URL
    }
  }

  /**
   * Handles paste event, checks for image in clipboard and uploads it.
   * @param {ClipboardEvent} event - Clipboard event object.
   */
  handlePaste(event) {
    const items = event.clipboardData.items;
    for (let i = 0; i < items.length; i++) {
      const item = items[i];
      if (item.type.indexOf("image") !== -1) {
        const file = item.getAsFile();
        this.handleFileUpload(file); // Handle image file upload
        break; // Only handle first image
      }
    }
  }

  /**
   * Uploads a base64 encoded image, sets it as canvas background.
   * @param {string} base64Image - Base64 encoded image data.
   */
  uploadBase64(base64Image) {
    // Check gradio version (likely version 4 or higher required).
    if (typeof this.gradio_config !== "undefined") {
      if (!this.gradio_config.version.startsWith("4")) {
        return;
      }
    } else {
      return;
    }

    const image = new Image();
    image.onload = () => {
      this.img = base64Image; // Set image data
      this.orgWidth = image.width; // Store original width
      this.orgHeight = image.height; // Store original height
      const canvas = document.getElementById("drawingCanvas_" + this.uuid);
      // Resize canvas to match image dimensions if different
      if (canvas.width !== image.width || canvas.height !== image.height) {
        canvas.width = image.width;
        canvas.height = image.height;
      }
      this.adjustInitialPositionAndScale(); // Adjust image position and scale
      this.drawImage(); // Draw the image on canvas
      this.on_img_upload(); // Trigger image upload callback
      this.saveState(); // Save canvas state to history
      document.getElementById("imageInput_" + this.uuid).value = null; // Clear file input
      document.getElementById("uploadHint_" + this.uuid).style.display = "none"; // Hide upload hint
    };

    // Set image source to base64 data or reset if no data
    if (base64Image) {
      image.src = base64Image;
    } else {
      this.img = null;
      const canvas = document.getElementById("drawingCanvas_" + this.uuid);
      canvas.width = 1; // Reset canvas dimensions if no image
      canvas.height = 1;
      this.adjustInitialPositionAndScale();
      this.drawImage();
      this.on_img_upload();
      this.saveState();
    }
  }

  /**
   * Uploads the drawing canvas content as base64 data.
   * @param {string} base64DrawingCanvas - Base64 encoded canvas image data.
   */
  uploadBase64DrawingCanvas(base64DrawingCanvas) {
    const image = new Image();
    image.onload = () => {
      const canvas = document.getElementById("drawingCanvas_" + this.uuid);
      const ctx = canvas.getContext("2d");
      ctx.clearRect(0, 0, canvas.width, canvas.height); // Clear canvas
      ctx.drawImage(image, 0, 0); // Draw loaded image onto canvas
      this.saveState(); // Save canvas state to history
    };

    // Set image source to base64 data or clear canvas if no data
    if (base64DrawingCanvas) {
      image.src = base64DrawingCanvas;
    } else {
      const canvas = document.getElementById("drawingCanvas_" + this.uuid);
      const ctx = canvas.getContext("2d");
      ctx.clearRect(0, 0, canvas.width, canvas.height); // Clear canvas
      this.saveState(); // Save empty canvas state
    }
  }

  /**
   * Checks if coordinates are inside the image bounds on the canvas.
   * @param {number} x - X-coordinate.
   * @param {number} y - Y-coordinate.
   * @returns {boolean} - True if inside image, false otherwise.
   */
  isInsideImage(x, y) {
    const imageWidthScaled = this.orgWidth * this.imgScale;
    const imageHeightScaled = this.orgHeight * this.imgScale;
    return (
      x > this.imgX &&
      x < this.imgX + imageWidthScaled &&
      y > this.imgY &&
      y < this.imgY + imageHeightScaled
    );
  }

  /**
   * Draws the loaded image on the canvas, adjusting position and scale.
   */
  drawImage() {
    const imageElement = document.getElementById("image_" + this.uuid),
      drawingCanvasElement = document.getElementById(
        "drawingCanvas_" + this.uuid,
      );

    if (this.img) {
      const imageWidthScaled = this.orgWidth * this.imgScale;
      const imageHeightScaled = this.orgHeight * this.imgScale;
      imageElement.src = this.img; // Set image source
      imageElement.style.width = imageWidthScaled + "px"; // Set scaled width
      imageElement.style.height = imageHeightScaled + "px"; // Set scaled height
      imageElement.style.left = this.imgX + "px"; // Set image X position
      imageElement.style.top = this.imgY + "px"; // Set image Y position
      imageElement.style.display = "block"; // Show image
      drawingCanvasElement.style.width = imageWidthScaled + "px"; // Match canvas width
      drawingCanvasElement.style.height = imageHeightScaled + "px"; // Match canvas height
      drawingCanvasElement.style.left = this.imgX + "px"; // Match canvas X position
      drawingCanvasElement.style.top = this.imgY + "px"; // Match canvas Y position
    } else {
      imageElement.src = ""; // Clear image source if no image
      imageElement.style.display = "none"; // Hide image element
    }
  }

  /**
   * Adjusts the initial position and scale of the image to fit within the container.
   */
  adjustInitialPositionAndScale() {
    const imageContainerElement = document.getElementById(
      "imageContainer_" + this.uuid,
    );
    const containerWidth = imageContainerElement.clientWidth - 32; // Container width minus padding
    const containerHeight = imageContainerElement.clientHeight - 32; // Container height minus padding
    const scaleX = containerWidth / this.orgWidth; // Scale to fit width
    const scaleY = containerHeight / this.orgHeight; // Scale to fit height
    this.imgScale = Math.min(scaleX, scaleY); // Use smaller scale to fit within container

    const imageWidthScaled = this.orgWidth * this.imgScale;
    const imageHeightScaled = this.orgHeight * this.imgScale;
    this.imgX = (imageContainerElement.clientWidth - imageWidthScaled) / 2; // Center image horizontally
    this.imgY = (imageContainerElement.clientHeight - imageHeightScaled) / 2; // Center image vertically
  }

  /**
   * Resets the canvas, clearing scribbles and adjusting image position/scale.
   */
  resetImage() {
    const canvas = document.getElementById("drawingCanvas_" + this.uuid);
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height); // Clear canvas content
    this.adjustInitialPositionAndScale(); // Reset image position and scale
    this.drawImage(); // Redraw image
    this.saveState(); // Save empty canvas state
  }

  /**
   * Removes the image from the canvas, clearing scribbles and resetting UI.
   */
  removeImage() {
    this.img = null; // Remove image data
    const imageElement = document.getElementById("image_" + this.uuid),
      canvas = document.getElementById("drawingCanvas_" + this.uuid),
      ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height); // Clear canvas content
    imageElement.src = ""; // Clear image source
    imageElement.style.width = "0"; // Reset image width
    imageElement.style.height = "0"; // Reset image height
    this.saveState(); // Save empty state
    if (!this.no_upload)
      document.getElementById("uploadHint_" + this.uuid).style.display =
        "block"; // Show upload hint
    this.on_img_upload(); // Trigger image upload callback (empty image)
  }

  /**
   * Saves the current state of the drawing canvas to history for undo/redo.
   */
  saveState() {
    const canvas = document.getElementById("drawingCanvas_" + this.uuid);
    const ctx = canvas.getContext("2d");
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height); // Get current canvas image data
    this.history = this.history.slice(0, this.historyIndex + 1); // Truncate redo history
    this.history.push(imageData); // Add current state to history
    this.historyIndex++; // Increment history index
    this.updateUndoRedoButtons(); // Update undo/redo button states
    this.on_drawing_canvas_upload(); // Trigger drawing canvas upload callback
  }

  /**
   * Undoes the last drawing action by restoring the previous canvas state.
   */
  undo() {
    if (this.historyIndex > 0) {
      this.historyIndex--; // Decrement history index
      this.restoreState(); // Restore previous state
      this.updateUndoRedoButtons(); // Update undo/redo buttons
    }
  }

  /**
   * Redoes the last undone action by restoring the next canvas state in history.
   */
  redo() {
    if (this.historyIndex < this.history.length - 1) {
      this.historyIndex++; // Increment history index
      this.restoreState(); // Restore next state
      this.updateUndoRedoButtons(); // Update undo/redo buttons
    }
  }

  /**
   * Restores the canvas state from the history at the current history index.
   */
  restoreState() {
    const canvas = document.getElementById("drawingCanvas_" + this.uuid);
    const ctx = canvas.getContext("2d");
    const state = this.history[this.historyIndex]; // Get state from history
    ctx.putImageData(state, 0, 0); // Put image data onto canvas
    this.on_drawing_canvas_upload(); // Trigger drawing canvas upload callback
  }

  /**
   * Updates the disabled state and opacity of undo/redo buttons based on history.
   */
  updateUndoRedoButtons() {
    const undoButtonElement = document.getElementById(
        "undoButton_" + this.uuid,
      ),
      redoButtonElement = document.getElementById("redoButton_" + this.uuid);
    undoButtonElement.disabled = this.historyIndex <= 0; // Disable undo if no history
    redoButtonElement.disabled = this.historyIndex >= this.history.length - 1; // Disable redo if at newest state
    undoButtonElement.style.opacity = undoButtonElement.disabled ? "0.5" : "1"; // Grey out disabled undo
    redoButtonElement.style.opacity = redoButtonElement.disabled ? "0.5" : "1"; // Grey out disabled redo
  }

  /**
   * Callback function triggered after image upload, updates background gradio bind.
   */
  on_img_upload() {
    if (!this.img) {
      this.background_gradio_bind.setValue(""); // Clear background bind if no image
      return;
    }
    const imageElement = document.getElementById("image_" + this.uuid),
      tempCanvas = this.temp_canvas,
      tempCtx = tempCanvas.getContext("2d");
    tempCanvas.width = this.orgWidth; // Match temp canvas width
    tempCanvas.height = this.orgHeight; // Match temp canvas height
    tempCtx.drawImage(imageElement, 0, 0, this.orgWidth, this.orgHeight); // Draw image onto temp canvas
    const base64Data = tempCanvas.toDataURL("image/png"); // Get base64 data from temp canvas
    this.background_gradio_bind.setValue(base64Data); // Set background gradio bind value
  }

  /**
   * Callback function triggered after drawing canvas upload, updates foreground gradio bind.
   */
  on_drawing_canvas_upload() {
    if (!this.img) {
      this.foreground_gradio_bind.setValue(""); // Clear foreground bind if no image
      return;
    }
    const canvas = document.getElementById("drawingCanvas_" + this.uuid),
      base64Data = canvas.toDataURL("image/png"); // Get base64 data from drawing canvas
    this.foreground_gradio_bind.setValue(base64Data); // Set foreground gradio bind value
  }

  /**
   * Maximizes the canvas container to full screen.
   */
  maximize() {
    if (this.maximized) {
      return; // Prevent maximize if already maximized
    }
    const containerElement = document.getElementById("container_" + this.uuid),
      toolbarElement = document.getElementById("toolbar_" + this.uuid),
      maximizeButtonElement = document.getElementById("maxButton_" + this.uuid),
      minimizeButtonElement = document.getElementById("minButton_" + this.uuid);

    // Store original container styles before maximization
    this.originalState = {
      width: containerElement.style.width,
      height: containerElement.style.height,
      top: containerElement.style.top,
      left: containerElement.style.left,
      position: containerElement.style.position,
      zIndex: containerElement.style.zIndex,
    };

    // Set container styles for maximized state
    containerElement.style.width = "100vw";
    containerElement.style.height = "100vh";
    containerElement.style.top = "0";
    containerElement.style.left = "0";
    containerElement.style.position = "fixed";
    containerElement.style.zIndex = "1000";
    maximizeButtonElement.style.display = "none"; // Hide maximize button
    minimizeButtonElement.style.display = "inline-block"; // Show minimize button
    this.maximized = true; // Set maximized flag
  }

  /**
   * Minimizes the canvas container, restoring it to its original size and position.
   */
  minimize() {
    if (!this.maximized) {
      return; // Prevent minimize if not maximized
    }
    const containerElement = document.getElementById("container_" + this.uuid),
      maximizeButtonElement = document.getElementById("maxButton_" + this.uuid),
      minimizeButtonElement = document.getElementById("minButton_" + this.uuid);

    // Restore original container styles from stored state
    containerElement.style.width = this.originalState.width;
    containerElement.style.height = this.originalState.height;
    containerElement.style.top = this.originalState.top;
    containerElement.style.left = this.originalState.left;
    containerElement.style.position = this.originalState.position;
    containerElement.style.zIndex = this.originalState.zIndex;
    maximizeButtonElement.style.display = "inline-block"; // Show maximize button
    minimizeButtonElement.style.display = "none"; // Hide minimize button
    this.maximized = false; // Reset maximized flag
  }

  /**
   * Handles the end of a drag operation, resetting dragging flags and cursors.
   * @param {PointerEvent} event - Pointer event object.
   * @param {boolean} pointerLeave - True if drag ended due to pointer leaving container.
   */
  handleDragEnd(event, pointerLeave) {
    const imageElement = document.getElementById("image_" + this.uuid),
      drawingCanvasElement = document.getElementById(
        "drawingCanvas_" + this.uuid,
      );
    this.dragging = false; // Reset dragging flag
    imageElement.style.cursor = "grab"; // Reset image cursor
    drawingCanvasElement.style.cursor = "grab"; // Reset canvas cursor
  }
}

// TypeCorrection. See comment in canvas.js
const True = true,
  False = false;
