function toggleCss(key, css, enable) {
    var style = document.getElementById(key);
    if (enable && !style) {
        style = document.createElement('style');
        style.id = key;
        style.type = 'text/css';
        document.head.appendChild(style);
    }
    if (style && !enable) {
        document.head.removeChild(style);
    }
    if (style) {
        style.innerHTML == '';
        style.appendChild(document.createTextNode(css));
    }
}

function setupExtraNetworksForTab(tabname) {
    function registerPrompt(tabname, id) {
        var textarea = gradioApp().querySelector("#" + id + " > label > textarea");

        if (!activePromptTextarea[tabname]) {
            activePromptTextarea[tabname] = textarea;
        }

        textarea.addEventListener("focus", function() {
            activePromptTextarea[tabname] = textarea;
        });
    }

    var tabnav = gradioApp().querySelector('#' + tabname + '_extra_tabs > div.tab-nav');
    var controlsDiv = document.createElement('DIV');
    controlsDiv.classList.add('extra-networks-controls-div');
    tabnav.appendChild(controlsDiv);
    tabnav.insertBefore(controlsDiv, null);

    var this_tab = gradioApp().querySelector('#' + tabname + '_extra_tabs');
    this_tab.querySelectorAll(":scope > [id^='" + tabname + "_']").forEach(function(elem) {
        var tabname_full = elem.id;
        var search = gradioApp().querySelector("#" + tabname_full + "_extra_search");
        var sort_dir = gradioApp().querySelector("#" + tabname_full + "_extra_sort_dir");
        var refresh = gradioApp().querySelector("#" + tabname_full + "_extra_refresh");
        var currentSort = '';

        if (!search || !sort_dir || !refresh) {
            return;
        }

        var applyFilter = function(force) {
            var searchTerm = search.value.toLowerCase();
            gradioApp().querySelectorAll('#' + tabname + '_extra_tabs div.card').forEach(function(elem) {
                var searchOnly = elem.querySelector('.search_only');
                var text = Array.prototype.map.call(elem.querySelectorAll('.search_terms, .description'), function(t) {
                    return t.textContent.toLowerCase();
                }).join(" ");

                var visible = text.indexOf(searchTerm) != -1;
                if (searchOnly && searchTerm.length < 4) {
                    visible = false;
                }
                if (visible) {
                    elem.classList.remove("hidden");
                } else {
                    elem.classList.add("hidden");
                }
            });

            applySort(force);
        };

        var applySort = function(force) {
            var cards = gradioApp().querySelectorAll('#' + tabname_full + ' div.card');
            var parent = gradioApp().querySelector('#' + tabname_full + "_cards");
            var reverse = sort_dir.dataset.sortdir == "Descending";
            var activeSearchElem = gradioApp().querySelector('#' + tabname_full + "_controls .extra-network-control--sort.extra-network-control--enabled");
            var sortKey = activeSearchElem ? activeSearchElem.dataset.sortkey : "default";
            var sortKeyDataField = "sort" + sortKey.charAt(0).toUpperCase() + sortKey.slice(1);
            var sortKeyStore = sortKey + "-" + sort_dir.dataset.sortdir + "-" + cards.length;
    
            if (sortKeyStore == currentSort && !force) {
                return;
            }
            currentSort = sortKeyStore;
    
            var sortedCards = Array.from(cards);
            sortedCards.sort(function(cardA, cardB) {
                var a = cardA.dataset[sortKeyDataField];
                var b = cardB.dataset[sortKeyDataField];
                if (!isNaN(a) && !isNaN(b)) {
                    return parseInt(a) - parseInt(b);
                }
                return (a < b ? -1 : (a > b ? 1 : 0));
            });
    
            if (reverse) {
                sortedCards.reverse();
            }
    
            parent.innerHTML = '';
            var frag = document.createDocumentFragment();
            sortedCards.forEach(function(card) {
                frag.appendChild(card);
            });
            parent.appendChild(frag);
        };

        search.addEventListener("input", function() {
            applyFilter();
        });
        applySort();
        applyFilter();
        extraNetworksApplySort[tabname_full] = applySort;
        extraNetworksApplyFilter[tabname_full] = applyFilter;

        var controls = gradioApp().querySelector("#" + tabname_full + "_controls");
        controlsDiv.insertBefore(controls, null);

        if (elem.style.display != "none") {
            extraNetworksShowControlsForPage(tabname, tabname_full);
        }
    });

    registerPrompt(tabname, tabname + "_prompt");
    registerPrompt(tabname, tabname + "_neg_prompt");
}

function extraNetworksMovePromptToTab(tabname, id, showPrompt, showNegativePrompt) {
    if (!gradioApp().querySelector('.toprow-compact-tools')) return;

    var promptContainer = gradioApp().getElementById(tabname + '_prompt_container');
    var prompt = gradioApp().getElementById(tabname + '_prompt_row');
    var negPrompt = gradioApp().getElementById(tabname + '_neg_prompt_row');
    var elem = id ? gradioApp().getElementById(id) : null;

    if (showNegativePrompt && elem) {
        elem.insertBefore(negPrompt, elem.firstChild);
    } else {
        promptContainer.insertBefore(negPrompt, promptContainer.firstChild);
    }

    if (showPrompt && elem) {
        elem.insertBefore(prompt, elem.firstChild);
    } else {
        promptContainer.insertBefore(prompt, promptContainer.firstChild);
    }

    if (elem) {
        elem.classList.toggle('extra-page-prompts-active', showNegativePrompt || showPrompt);
    }
}

function extraNetworksShowControlsForPage(tabname, tabname_full) {
    gradioApp().querySelectorAll('#' + tabname + '_extra_tabs .extra-networks-controls-div > div').forEach(function(elem) {
        var targetId = tabname_full + "_controls";
        elem.style.display = elem.id == targetId ? "" : "none";
    });
}

function extraNetworksUnrelatedTabSelected(tabname) {
    extraNetworksMovePromptToTab(tabname, '', false, false);
    extraNetworksShowControlsForPage(tabname, null);
}

function extraNetworksTabSelected(tabname, id, showPrompt, showNegativePrompt, tabname_full) {
    extraNetworksMovePromptToTab(tabname, id, showPrompt, showNegativePrompt);
    extraNetworksShowControlsForPage(tabname, tabname_full);
}

function applyExtraNetworkFilter(tabname_full) {
    var doFilter = function() {
        var applyFunction = extraNetworksApplyFilter[tabname_full];
        if (applyFunction) {
            applyFunction(true);
        }
    };
    setTimeout(doFilter, 1);
}

function applyExtraNetworkSort(tabname_full) {
    var doSort = function() {
        extraNetworksApplySort[tabname_full](true);
    };
    setTimeout(doSort, 1);
}

var extraNetworksApplyFilter = {};
var extraNetworksApplySort = {};
var activePromptTextarea = {};

function setupExtraNetworks() {
    setupExtraNetworksForTab('txt2img');
    setupExtraNetworksForTab('img2img');
}

var re_extranet = /<([^:^>]+:[^:]+):[\d.]+>(.*)/s;
var re_extranet_g = /<([^:^>]+:[^:]+):[\d.]+>/g;
var re_extranet_neg = /\(([^:^>]+:[\d.]+)\)/;
var re_extranet_g_neg = /\(([^:^>]+:[\d.]+)\)/g;

function tryToRemoveExtraNetworkFromPrompt(textarea, text, isNeg) {
    var m = text.match(isNeg ? re_extranet_neg : re_extranet);
    var replaced = false;
    var newTextareaText;
    var extraTextBeforeNet = opts.extra_networks_add_text_separator;
    if (m) {
        var extraTextAfterNet = m[2];
        var partToSearch = m[1];
        var foundAtPosition = -1;
        newTextareaText = textarea.value.replaceAll(isNeg ? re_extranet_g_neg : re_extranet_g, function(found, net, pos) {
            m = found.match(isNeg ? re_extranet_neg : re_extranet);
            if (m[1] == partToSearch) {
                replaced = true;
                foundAtPosition = pos;
                return "";
            }
            return found;
        });
        if (foundAtPosition >= 0) {
            if (extraTextAfterNet && newTextareaText.substr(foundAtPosition, extraTextAfterNet.length) == extraTextAfterNet) {
                newTextareaText = newTextareaText.substr(0, foundAtPosition) + newTextareaText.substr(foundAtPosition + extraTextAfterNet.length);
            }
            if (newTextareaText.substr(foundAtPosition - extraTextBeforeNet.length, extraTextBeforeNet.length) == extraTextBeforeNet) {
                newTextareaText = newTextareaText.substr(0, foundAtPosition - extraTextBeforeNet.length) + newTextareaText.substr(foundAtPosition);
            }
        }
    } else {
        newTextareaText = textarea.value.replaceAll(new RegExp(`((?:${extraTextBeforeNet})?${text})`, "g"), "");
        replaced = (newTextareaText != textarea.value);
    }

    if (replaced) {
        textarea.value = newTextareaText;
        return true;
    }

    return false;
}

function updatePromptArea(text, textArea, isNeg) {
    if (!tryToRemoveExtraNetworkFromPrompt(textArea, text, isNeg)) {
        textArea.value = textArea.value + opts.extra_networks_add_text_separator + text;
    }

    updateInput(textArea);
}

function cardClicked(tabname, textToAdd, textToAddNegative, allowNegativePrompt) {
    if (textToAddNegative.length > 0) {
        updatePromptArea(textToAdd, gradioApp().querySelector("#" + tabname + "_prompt > label > textarea"));
        updatePromptArea(textToAddNegative, gradioApp().querySelector("#" + tabname + "_neg_prompt > label > textarea"), true);
    } else {
        var textarea = allowNegativePrompt ? activePromptTextarea[tabname] : gradioApp().querySelector("#" + tabname + "_prompt > label > textarea");
        updatePromptArea(textToAdd, textarea);
    }
}

function saveCardPreview(event, tabname, filename) {
    var textarea = gradioApp().querySelector("#" + tabname + '_preview_filename  > label > textarea');
    var button = gradioApp().getElementById(tabname + '_save_preview');

    textarea.value = filename;
    updateInput(textarea);

    button.click();

    event.stopPropagation();
    event.preventDefault();
}

function extraNetworksSearchButton(tabname, extra_networks_tabname, event) {
    var searchTextarea = gradioApp().querySelector("#" + tabname + "_" + extra_networks_tabname + "_extra_search");
    var button = event.target;
    var text = button.classList.contains("search-all") ? "" : button.textContent.trim();

    searchTextarea.value = text;
    updateInput(searchTextarea);
}

function extraNetworksTreeProcessDirectoryClick(event, btn, tabname, extra_networks_tabname) {
    var ul = btn.nextElementSibling;
    var true_targ = event.target;

    function _expand_or_collapse(_ul, _btn) {
        if (_ul.hasAttribute("hidden")) {
            _ul.removeAttribute("hidden");
            _btn.dataset.expanded = "";
        } else {
            _ul.setAttribute("hidden", "");
            delete _btn.dataset.expanded;
        }
    }

    function _remove_selected_from_all() {
        var sels = document.querySelectorAll("div.tree-list-content");
        [...sels].forEach(el => {
            delete el.dataset.selected;
        });
    }

    function _select_button(_btn) {
        _remove_selected_from_all();
        _btn.dataset.selected = "";
    }

    function _update_search(_tabname, _extra_networks_tabname, _search_text) {
        var search_input_elem = gradioApp().querySelector("#" + tabname + "_" + extra_networks_tabname + "_extra_search");
        search_input_elem.value = _search_text;
        updateInput(search_input_elem);
    }

    if (true_targ.matches(".tree-list-item-action--leading, .tree-list-item-action-chevron")) {
        _expand_or_collapse(ul, btn);
    } else {
        if ("selected" in btn.dataset && !(ul.hasAttribute("hidden"))) {
            _expand_or_collapse(ul, btn);
            delete btn.dataset.selected;
            _update_search(tabname, extra_networks_tabname, "");
        } else if (!(!("selected" in btn.dataset) && !(ul.hasAttribute("hidden")))) {
            _expand_or_collapse(ul, btn);
            _select_button(btn, tabname, extra_networks_tabname);
            _update_search(tabname, extra_networks_tabname, btn.dataset.path);
        } else {
            _select_button(btn, tabname, extra_networks_tabname);
            _update_search(tabname, extra_networks_tabname, btn.dataset.path);
        }
    }
}

function extraNetworksTreeOnClick(event, tabname, extra_networks_tabname) {
    var btn = event.currentTarget;
    var par = btn.parentElement;
    if (par.dataset.treeEntryType === "file") {
        extraNetworksTreeProcessFileClick(event, btn, tabname, extra_networks_tabname);
    } else {
        extraNetworksTreeProcessDirectoryClick(event, btn, tabname, extra_networks_tabname);
    }
}

function extraNetworksControlSortOnClick(event, tabname, extra_networks_tabname) {
    var self = event.currentTarget;
    var parent = event.currentTarget.parentElement;

    parent.querySelectorAll('.extra-network-control--sort').forEach(function(x) {
        x.classList.remove('extra-network-control--enabled');
    });

    self.classList.add('extra-network-control--enabled');

    applyExtraNetworkSort(tabname + "_" + extra_networks_tabname);
}

function extraNetworksControlSortDirOnClick(event, tabname, extra_networks_tabname) {
    if (event.currentTarget.dataset.sortdir == "Ascending") {
        event.currentTarget.dataset.sortdir = "Descending";
        event.currentTarget.setAttribute("title", "Sort descending");
    } else {
        event.currentTarget.dataset.sortdir = "Ascending";
        event.currentTarget.setAttribute("title", "Sort ascending");
    }
    applyExtraNetworkSort(tabname + "_" + extra_networks_tabname);
}

function extraNetworksControlTreeViewOnClick(event, tabname, extra_networks_tabname) {
    var button = event.currentTarget;
    button.classList.toggle("extra-network-control--enabled");
    
    var pane = gradioApp().getElementById(tabname + "_" + extra_networks_tabname + "_pane");
    pane.classList.toggle("extra-network-dirs-hidden");

    var tree = gradioApp().getElementById(tabname + "_" + extra_networks_tabname + "_tree");
    var cards = gradioApp().getElementById(tabname + "_" + extra_networks_tabname + "_cards");
    var resizeHandle = pane.querySelector('.resize-handle');

    if (pane.classList.contains("extra-network-dirs-hidden")) {
        tree.style.display = 'none';
        cards.style.flexGrow = '1';
        cards.style.width = '100%';
        if (resizeHandle) resizeHandle.style.display = 'none';
    } else {
        tree.style.display = 'block';
        cards.style.flexGrow = '1';
        cards.style.width = 'auto';
        if (resizeHandle) resizeHandle.style.display = 'block';
    }
}

function extraNetworksSearchButton(tabname, extra_networks_tabname, event) {
    var searchTextarea = gradioApp().querySelector("#" + tabname + "_" + extra_networks_tabname + "_extra_search");
    var button = event.target;
    var text = button.classList.contains("search-all") ? "" : button.textContent.trim();

    searchTextarea.value = text;
    updateInput(searchTextarea);
}

function extraNetworksControlRefreshOnClick(event, tabname, extra_networks_tabname) {
    var btn_refresh_internal = gradioApp().getElementById(tabname + "_" + extra_networks_tabname + "_extra_refresh_internal");
    btn_refresh_internal.dispatchEvent(new Event("click"));
}

var globalPopup = null;
var globalPopupInner = null;

function closePopup() {
    if (!globalPopup) return;
    globalPopup.style.display = "none";
}

function popup(contents) {
    if (!globalPopup) {
        globalPopup = document.createElement('div');
        globalPopup.classList.add('global-popup');

        var close = document.createElement('div');
        close.classList.add('global-popup-close');
        close.addEventListener("click", closePopup);
        close.title = "Close";
        globalPopup.appendChild(close);

        globalPopupInner = document.createElement('div');
        globalPopupInner.classList.add('global-popup-inner');
        globalPopup.appendChild(globalPopupInner);

        gradioApp().querySelector('.main').appendChild(globalPopup);
    }

    globalPopupInner.innerHTML = '';
    globalPopupInner.appendChild(contents);

    globalPopup.style.display = "flex";
}

var storedPopupIds = {};
function popupId(id) {
    if (!storedPopupIds[id]) {
        storedPopupIds[id] = gradioApp().getElementById(id);
    }

    popup(storedPopupIds[id]);
}

function extraNetworksFlattenMetadata(obj) {
    const result = {};

    for (const key of Object.keys(obj)) {
        if (typeof obj[key] === 'string') {
            try {
                const parsed = JSON.parse(obj[key]);
                if (parsed && typeof parsed === 'object') {
                    obj[key] = parsed;
                }
            } catch (error) {
                continue;
            }
        }
    }

    for (const key of Object.keys(obj)) {
        if (typeof obj[key] === 'object' && obj[key] !== null) {
            const nested = extraNetworksFlattenMetadata(obj[key]);
            for (const nestedKey of Object.keys(nested)) {
                result[`${key}/${nestedKey}`] = nested[nestedKey];
            }
        } else {
            result[key] = obj[key];
        }
    }

    for (const key of Object.keys(result)) {
        if (key.startsWith("modelspec.")) {
            result[key.replaceAll(".", "/")] = result[key];
            delete result[key];
        }
    }

    for (const key of Object.keys(result)) {
        const parts = key.split("/");
        for (let i = 1; i < parts.length; i++) {
            const parent = parts.slice(0, i).join("/");
            if (!result[parent]) {
                result[parent] = "";
            }
        }
    }

    return result;
}

function extraNetworksShowMetadata(text) {
    try {
        let parsed = JSON.parse(text);
        if (parsed && typeof parsed === 'object') {
            parsed = extraNetworksFlattenMetadata(parsed);
            const table = createVisualizationTable(parsed, 0);
            popup(table);
            return;
        }
    } catch (error) {
        console.error(error);
    }
    
    var elem = document.createElement('pre');
    elem.classList.add('popup-metadata');
    elem.textContent = text;

    popup(elem);
}

function requestGet(url, data, handler, errorHandler) {
    var xhr = new XMLHttpRequest();
    var args = Object.keys(data).map(function(k) {
        return encodeURIComponent(k) + '=' + encodeURIComponent(data[k]);
    }).join('&');
    xhr.open("GET", url + "?" + args, true);

    xhr.onreadystatechange = function() {
        if (xhr.readyState === 4) {
            if (xhr.status === 200) {
                try {
                    var js = JSON.parse(xhr.responseText);
                    handler(js);
                } catch (error) {
                    console.error(error);
                    errorHandler();
                }
            } else {
                errorHandler();
            }
        }
    };
    var js = JSON.stringify(data);
    xhr.send(js);
}

function extraNetworksCopyCardPath(event) {
    navigator.clipboard.writeText(event.target.getAttribute("data-clipboard-text"));
    event.stopPropagation();
}

function extraNetworksRequestMetadata(event, extraPage) {
    var showError = function() {
        extraNetworksShowMetadata("there was an error getting metadata");
    };

    var cardName = event.target.parentElement.parentElement.getAttribute("data-name") || event.target.parentElement.parentElement.parentElement.getAttribute("data-name");

    requestGet("./sd_extra_networks/metadata", {page: extraPage, item: cardName}, function(data) {
        if (data && data.metadata) {
            extraNetworksShowMetadata(data.metadata);
        } else {
            showError();
        }
    }, showError);

    event.stopPropagation();
}

var extraPageUserMetadataEditors = {};

function extraNetworksEditUserMetadata(event, tabname, extraPage) {
    var id = tabname + '_' + extraPage + '_edit_user_metadata';

    var editor = extraPageUserMetadataEditors[id];
    if (!editor) {
        editor = {};
        editor.page = gradioApp().getElementById(id);
        editor.nameTextarea = gradioApp().querySelector("#" + id + "_name" + ' textarea');
        editor.button = gradioApp().querySelector("#" + id + "_button");
        extraPageUserMetadataEditors[id] = editor;
    }
    
    var cardName = event.target.parentElement.parentElement.getAttribute("data-name") || event.target.parentElement.parentElement.parentElement.getAttribute("data-name");

    editor.nameTextarea.value = cardName;
    updateInput(editor.nameTextarea);

    editor.button.click();

    popup(editor.page);

    event.stopPropagation();
}

function extraNetworksRefreshSingleCard(page, tabname, name) {
    requestGet("./sd_extra_networks/get-single-card", {page: page, tabname: tabname, name: name}, function(data) {
        if (data && data.html) {
            var card = gradioApp().querySelector(`#${tabname}_${page.replace(" ", "_")}_cards > .card[data-name="${name}"]`);

            var newDiv = document.createElement('DIV');
            newDiv.innerHTML = data.html;
            var newCard = newDiv.firstElementChild;

            newCard.style.display = '';
            card.parentElement.insertBefore(newCard, card);
            card.parentElement.removeChild(card);
        }
    });
}

window.addEventListener("keydown", function(event) {
    if (event.key == "Escape") {
        closePopup();
    }
});

var uiAfterScriptsCallbacks = [];
var uiAfterScriptsTimeout = null;
var executedAfterScripts = false;

function scheduleAfterScriptsCallbacks() {
    clearTimeout(uiAfterScriptsTimeout);
    uiAfterScriptsTimeout = setTimeout(function() {
        executeCallbacks(uiAfterScriptsCallbacks);
    }, 200);
}

onUiLoaded(function() {
    var mutationObserver = new MutationObserver(function(m) {
        let existingSearchfields = gradioApp().querySelectorAll("[id$='_extra_search']").length;
        let neededSearchfields = gradioApp().querySelectorAll("[id$='_extra_tabs'] > .tab-nav > button").length - 2;

        if (!executedAfterScripts && existingSearchfields >= neededSearchfields) {
            mutationObserver.disconnect();
            executedAfterScripts = true;
            scheduleAfterScriptsCallbacks();
        }
    });
    mutationObserver.observe(gradioApp(), {childList: true, subtree: true});
});

uiAfterScriptsCallbacks.push(setupExtraNetworks);